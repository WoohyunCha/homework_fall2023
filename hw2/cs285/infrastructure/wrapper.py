import gym

class VectorTimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_timesteps, ob_dim, ac_dim):
        super().__init__(env)
        self.max_timesteps = max_timesteps
        self.timesteps = [0] * self.num_envs
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim

    def reset(self, **kwargs):
        self.timesteps = [0] * self.num_envs
        return self.env.reset(**kwargs)

    def step(self, actions):
        # print(self.env.step(actions))
        observations, rewards, dones, truncated, infos = self.env.step(actions)
        self.timesteps = [t + 1 for t in self.timesteps]
        for i, done in enumerate(dones):
            if self.timesteps[i] >= self.max_timesteps:
                dones[i] = True
                truncated[i] = True
        return observations, rewards, dones, truncated