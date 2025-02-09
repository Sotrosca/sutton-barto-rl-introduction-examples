
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