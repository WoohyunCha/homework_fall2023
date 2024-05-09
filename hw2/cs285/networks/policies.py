import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
import math

class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )
        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        dist = self(ptu.from_numpy(obs))
        if self.discrete:
            # print(dist)
            # if math.isnan(dist[0]):
            #     print(obs)
            #     print(self(ptu.from_numpy(obs)))
            action = torch.multinomial(torch.exp(dist), num_samples=1).item()
        else:
            action = ptu.to_numpy(dist.rsample())
        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space. DONE
            ret = F.log_softmax(self.logits_net(obs), dim=-1)
        else:
            # TODO: define the forward pass for a policy with a continuous action space. DONE
            ret = distributions.Normal(self.mean_net(obs), torch.exp(self.logstd))
        return ret

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        self.optimizer.zero_grad()
        criterion = torch.nn.MSELoss()
        loss = criterion(self(obs), actions)
        loss.backward()
        self.optimizer.step()
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
    
    def get_logprob(self, obs: torch.FloatTensor, actions: torch.FloatTensor):
        dist = self(obs)
        if self.discrete:
            logprob = dist[torch.arange(0, dist.size(0)),actions.long()]
        else:
            logprob = dist.log_prob(actions)
        return logprob.requires_grad_(True)


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        logprob = self.get_logprob(obs, actions)
        self.optimizer.zero_grad()
        loss = -torch.sum(logprob.view((advantages.size(0),-1)) * advantages.view((-1,1))) # - for gradient descent
        loss.backward()
        self.optimizer.step()
        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
