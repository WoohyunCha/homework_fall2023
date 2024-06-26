from collections import OrderedDict
import numpy as np
import copy
from cs285.networks.policies import MLPPolicy
import gym
import cv2
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.pytorch_util import from_numpy, to_numpy
from typing import Dict, Tuple, List

############################################
############################################


def sample_trajectory(
    env: gym.Env, policy: MLPPolicy, max_length: int, render: bool = False
) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy."""
    ob, _ = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render an image
        # if render:
        #     if hasattr(env, "sim"):
        #         img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
        #     else:
        #         img = env.render(mode="single_rgb_array")
        #         # img = env.render()
        #     image_obs.append(
        #         cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
        #     )

        # TODO use the most recent ob and the policy to decide what to do
        ac =  policy.get_action(ob)
        # TODO: use that action to take a step in the environment
        next_ob, rew, done, _, _ = env.step(ac)

        # TODO rollout can end due to done, or due to max_length
        steps += 1
        rollout_done = done | (steps > max_length)

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def sample_trajectories(
    env: gym.Env,
    policy: MLPPolicy,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)

        # count steps
        timesteps_this_batch += get_traj_length(traj)
    return trajs, timesteps_this_batch



def sample_n_trajectories(
    env: gym.Env, policy: MLPPolicy, ntraj: int, max_length: int, render: bool = False
):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs


def compute_metrics(trajs, eval_trajs):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [traj["reward"].sum() for traj in trajs]
    eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]

    # episode lengths, for logging
    train_ep_lens = [len(traj["reward"]) for traj in trajs]
    eval_ep_lens = [len(eval_traj["reward"]) for eval_traj in eval_trajs]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


def convert_listofrollouts(trajs):
    """
    Take a list of rollout dictionaries and return separate arrays, where each array is a concatenation of that array
    from across the rollouts.
    """
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    concatenated_rewards = np.concatenate([traj["reward"] for traj in trajs])
    unconcatenated_rewards = [traj["reward"] for traj in trajs]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )


def get_traj_length(traj):
    return len(traj["reward"])

################MULTITHREAD#############

def sample_trajectories_vectorize(
    env: gym.Env,
    policy: MLPPolicy,
    min_timesteps_per_batch: int,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    trajs = []
    timesteps_this_batch = 0
    ob = env.reset()[0]
    obs, acs, rewards, next_obs, terminals = np.zeros((env.observation_space.shape[0],  min_timesteps_per_batch, env.observation_space.shape[1])),\
        np.zeros((env.observation_space.shape[0],  min_timesteps_per_batch, env.ac_dim)),\
            np.zeros((env.observation_space.shape[0], min_timesteps_per_batch)),\
                np.zeros((env.observation_space.shape[0],  min_timesteps_per_batch, env.observation_space.shape[1])),\
                np.zeros((env.observation_space.shape[0],  min_timesteps_per_batch)),
    step = 0
    reset_step = np.zeros(env.num_envs, dtype=int)
    while timesteps_this_batch < min_timesteps_per_batch:
        ac =  policy.get_action(ob)
        next_ob, rew, done, _ = env.step(ac)
        # record result of taking that action
        obs[:, step, :] = ob
        acs[:, step, :] = ac
        next_obs[:, step, :] = next_ob
        terminals[:, step] = done
        rewards[:, step] = rew

        ob = next_ob  # jump to next timestep
        step += 1
        # check for done envs
        done_envs_index = np.where(done)[0]
        if done_envs_index.size:
            for done_index in done_envs_index:
                traj = {
                    "observation": obs[done_index, reset_step[done_index]:step],
                    "reward": rewards[done_index, reset_step[done_index]:step],
                    "action": acs[done_index, reset_step[done_index]:step],
                    "next_observation": next_obs[done_index, reset_step[done_index]:step],
                    "terminal": terminals[done_index, reset_step[done_index]:step]
                }
                trajs.append(traj)
                timesteps_this_batch += get_traj_length(traj)
                reset_step[done_index] = step
    return trajs, timesteps_this_batch

def sample_n_trajectories_vectorize(
    env: gym.Env, policy: MLPPolicy, ntraj: int, render: bool = False
):
    trajs = []
    timesteps_this_batch = 0
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals = np.zeros((env.observation_space.shape[0],  env.max_timesteps, env.observation_space.shape[1])),\
        np.zeros((env.action_space.shape[0],  env.max_timesteps, env.action_space.shape[1])),\
            np.zeros((env.observation_space.shape[0],  env.max_timesteps, 1)),\
                np.zeros((env.observation_space.shape[0],  env.max_timesteps, env.observation_space.shape[1])),\
                np.zeros((env.observation_space.shape[0],  env.max_timesteps, 1)),
    image_obs = []
    step = 0
    reset_step = np.zeros(env.num_envs)
    while True:
        if render:
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render(mode="single_rgb_array")
                # img = env.render()
            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )
        ac =  policy.get_action(ob)
        next_ob, rew, done, _ = env.step(ac)
        # record result of taking that action
        obs[:, step, :] = ob
        acs[:, step, :] = ac
        next_obs[:, step, :] = next_ob
        terminals[:, step] = done
        rewards[:, step] = rewards

        ob = next_ob  # jump to next timestep
        step += 1
        # check for done envs
        done_envs_index = np.where(done)[0]
        if done_envs_index.size:
            for done_index in done_envs_index:
                traj = {
                    "observation": obs[done_index, reset_step[done_index]:step],
                    "reward": rewards[done_index, reset_step[done_index]:step],
                    "action": acs[done_index, reset_step[done_index]:step],
                    "next_observation": next_obs[done_index, reset_step[done_index]:step],
                    "terminal": terminals[done_index, reset_step[done_index]:step],
                    "image obs": np.array(image_obs).swapaxes(0,1)[done_index, reset_step[done_index]:step]
                }
                trajs.append(traj)
                timesteps_this_batch += get_traj_length(traj)
                reset_step[done_index] = step
        if (len(trajs) >= ntraj):
            return trajs[:ntraj] 


