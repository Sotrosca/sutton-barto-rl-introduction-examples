import numpy as np


class NArmedBandit:
    def __init__(
        self,
        n,
        rewards=None,
        mean_rewards=None,
        variance_rewards=None,
        noise=0,
        **kwargs
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
            self.rewards = rewards
        elif mean_rewards is not None and variance_rewards is not None:
            self.mean_rewards = mean_rewards
            self.variance_rewards = variance_rewards
            self.rewards = mean_rewards
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


class NArmedBanditNonStationary(NArmedBandit):
    def __init__(
        self,
        n,
        rewards=None,
        noise=0,
        steps_to_change=100,
        set_rewards_func=None,
    ):
        self.n = n
        self.rewards = rewards
        self.action_count = np.zeros(n)
        self.time = 0
        self.last_action = None
        self.last_reward = None
        self.history = []
        self.noise = noise
        self.steps_to_change = steps_to_change
        self.set_rewards_func = set_rewards_func

    def _set_rewards(self, rewards):
        rewards_shuffle = np.random.permutation(rewards)
        self.rewards = rewards_shuffle

    def step(self, action):
        reward = self.rewards[action] + np.random.normal(0, self.noise)
        self.action_count[action] += 1
        self.time += 1
        self.last_action = action
        self.last_reward = reward
        self.history.append((action, reward))
        if self.time % self.steps_to_change == 0:
            self._set_rewards(rewards=self.rewards)
        return reward


class NArmedBanditRandomWalkRewardsUpdate(NArmedBanditNonStationary):
    def __init__(
        self,
        n,
        rewards=None,
        noise=0,
        steps_to_change=100,
        set_rewards_func=None,
    ):
        super().__init__(n, rewards, noise, steps_to_change, set_rewards_func)

    def _set_rewards(self, rewards):
        self.rewards += np.random.normal(0, 1, self.n)


class Agent:
    def __init__(self, env: NArmedBandit, epsilon=0.1, initial_q_values=0):
        self.env = env
        self.n = env.n
        self.q_values = np.zeros(self.n) + initial_q_values
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
    def __init__(self, env: NArmedBandit, epsilon=0.1, initial_q_values=0):
        super().__init__(env, epsilon, initial_q_values)

    def action_value_function(self, action):
        q_action = self.q_values[action]
        action_count = self.action_count[action]
        action_rewards = self.action_rewards[action]
        last_reward = action_rewards[-1]

        return q_action + (last_reward - q_action) / action_count


class UCBAgent(IncrementalAgent):
    def __init__(self, env: NArmedBandit, c):
        super().__init__(env)
        self.c = c

    def act(self, env):
        # Calculate the UCB
        def ucb(n, q, c, t):
            return q + c * np.sqrt(np.log(t) / n) if n > 0 else float("inf")

        t = self.env.time + 1
        c = self.c
        ucb_values = [ucb(n, q, c, t) for n, q in zip(self.action_count, self.q_values)]
        action = np.argmax(ucb_values)
        self.actions.append(action)
        return action


class GradientBanditAgent(IncrementalAgent):
    def __init__(self, env: NArmedBandit, alpha=0.1, baseline=True):
        super().__init__(
            env,
            epsilon=0,
            initial_q_values=0,
        )
        self.alpha = alpha
        self.baseline = baseline
        self.preferences = np.zeros(self.n)  # H_t(a)
        self.action_probabilities = np.zeros(self.n)
        self.average_reward = 0

    def act(self, env):
        self.action_probabilities = np.exp(self.preferences) / np.sum(
            np.exp(self.preferences)
        )  # π_t(a)
        action = np.random.choice(self.n, p=self.action_probabilities)
        self.actions.append(action)
        return action

    def update(self, action, reward):
        self.action_count[action] += 1
        self.rewards.append(reward)
        self.action_rewards[action].append(reward)
        self.q_values[action] = self.action_value_function(action)
        time = self.env.time
        self.average_reward = (
            (time - 1) / time * self.average_reward + reward / time
            if self.baseline
            else 0
        )
        for a in range(self.n):
            if a == action:
                self.preferences[a] += (
                    self.alpha
                    * (reward - self.average_reward)
                    * (1 - self.action_probabilities[a])
                )
            else:
                self.preferences[a] -= (
                    self.alpha
                    * (reward - self.average_reward)
                    * self.action_probabilities[a]
                )


class NonStationaryIncrementalAgent(Agent):
    def __init__(self, env: NArmedBandit, epsilon=0.1, alpha=0.1, initial_q_values=0):
        super().__init__(env, epsilon, initial_q_values)
        self.alpha = alpha

    def action_value_function(self, action):
        q_action = self.q_values[action]
        action_rewards = self.action_rewards[action]
        last_reward = action_rewards[-1]

        return q_action + self.alpha * (last_reward - q_action)
