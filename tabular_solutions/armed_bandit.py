import numpy as np


class NArmedBandit:
    def __init__(
        self, n, rewards=None, mean_rewards=None, variance_rewards=None, noise=0
    ):
        self.n = n
        self._set_rewards(rewards, mean_rewards, variance_rewards)
        self.noise = noise
        self.action_count = np.zeros(n)
        self.time = 0
        self.last_action = None
        self.last_reward = None
        self.history = []

    def _set_rewards(self, rewards, mean_rewards, variance_rewards):
        if rewards is not None:
            self.mean_rewards = rewards
            self.variance_rewards = np.zeros(self.n)
        elif mean_rewards is not None and variance_rewards is not None:
            self.mean_rewards = mean_rewards
            self.variance_rewards = variance_rewards
        else:
            raise ValueError(
                "Either rewards or mean_rewards and variance_rewards must be provided"
            )

    def reset(self):
        self.action_count = np.zeros(self.n)
        self.time = 0
        self.last_action = None
        self.history = []

    def step(self, action):
        reward = np.random.normal(
            self.mean_rewards[action], self.variance_rewards[action]
        ) + np.random.normal(0, self.noise)
        self.action_count[action] += 1
        self.time += 1
        self.last_action = action
        self.last_reward = reward
        self.history.append((action, reward))
        return reward


class Agent:
    def __init__(self, env: NArmedBandit, epsilon=0.1):
        self.env = env
        self.n = env.n
        self.q_values = np.zeros(self.n)
        self.action_count = np.zeros(self.n)
        self.time = 0
        self.last_action = None
        self.actions = []
        self.rewards = []
        self.action_rewards = [[] for _ in range(self.n)]
        self.epsilon = epsilon

    def action_value_function(self, action):
        return np.mean(self.action_rewards[action])

    def update(self, action, reward):
        self.action_count[action] += 1
        self.rewards.append(reward)
        self.action_rewards[action].append(reward)
        self.q_values[action] = self.action_value_function(action)

    def act(self, env):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.n)
        else:
            all_max_actions = np.where(self.q_values == np.max(self.q_values))[0]
            action = np.random.choice(all_max_actions)
        self.actions.append(action)
        return action

    def task(self, steps):
        for _ in range(steps):
            action = self.act(self.env)
            reward = self.env.step(action)
            self.update(action, reward)


class IncrementalAgent(Agent):
    def __init__(self, env: NArmedBandit, epsilon=0.1):
        super().__init__(env, epsilon)

    def action_value_function(self, action):
        q_action = self.q_values[action]
        action_count = self.action_count[action]
        action_rewards = self.action_rewards[action]
        last_reward = action_rewards[-1]

        return q_action + (last_reward - q_action) / action_count
