import numpy as np


class Arm:
    def __init__(self, p):
        self.p = p

    def pull(self):
        return np.random.random() < self.p


class Base_Agent:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms

    def choose_arm(self):
        pass

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = (((n - 1)) * value + reward) / float(n)
        self.values[arm] = new_value


class Greedy_Agent(Base_Agent):
    def __init__(self, n_arms):
        super().__init__(n_arms)

    def choose_arm(self):
        return np.argmax(self.values)


class Epsilon_Greedy_Agent(Base_Agent):
    def __init__(self, n_arms, epsilon):
        super().__init__(n_arms)
        self.epsilon = epsilon

    def choose_arm(self):
        if np.random.random() > self.epsilon:
            return np.argmax(self.values)
        else:
            return np.random.randint(self.n_arms)


class UCB_Agent(Base_Agent):
    def __init__(self, n_arms):
        super().__init__(n_arms)

    def choose_arm(self):
        n_arms = len(self.counts)
        total_counts = sum(self.counts)
        ucb_values = [0.0 for arm in range(n_arms)]
        for arm in range(n_arms):
            bonus = np.sqrt(
                (2 * np.log(total_counts)) / (float(self.counts[arm]) + 1e-8)
            )
            ucb_values[arm] = self.values[arm] + bonus
        return np.argmax(ucb_values)


class Thompson_Agent(Base_Agent):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.alpha = [1.0] * n_arms
        self.beta = [1.0] * n_arms

    def choose_arm(self):
        samples = [
            np.random.beta(self.alpha[arm], self.beta[arm])
            for arm in range(self.n_arms)
        ]
        return np.argmax(samples)

    def update(self, arm, reward):
        super().update(arm, reward)
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


results = np.zeros((9, 9, 4))


for a_idx, a_win in enumerate(np.arange(0.1, 1, 0.1)):
    for b_idx, b_win in enumerate(np.arange(0.1, 1, 0.1)):
        arms = [Arm(a_win), Arm(b_win)]
        g_a = Greedy_Agent(2)
        e_g_a = Epsilon_Greedy_Agent(2, 0.2)
        u_a = UCB_Agent(2)
        t_a = Thompson_Agent(2)
        agents = [g_a, e_g_a, u_a, t_a]
        for agent_idx, agent in enumerate(agents):
            for i in range(5):
                total_reward = 0
                for n in range(100):
                    if n == 0:
                        chosen_arm = np.random.randint(2)
                    else:
                        chosen_arm = agent.choose_arm()
                    reward = arms[chosen_arm].pull()
                    agent.update(chosen_arm, reward)
                    total_reward += reward
                results[a_idx, b_idx, agent_idx] += total_reward / 5

import matplotlib.pyplot as plt

probabilities = np.arange(0.1, 1, 0.1)
PA, PB = np.meshgrid(probabilities, probabilities)

plt.figure(figsize=(15, 15))

algorithms = ["Greedy", "Epsilon-Greedy", "UCB", "Thompson Sampling"]
colors = ["red", "blue", "green", "purple"]

for idx, (algo, color) in enumerate(zip(algorithms, colors)):
    print(f"algorithm: {algo}")
    print(f"total reward: {np.sum(results[:, :, idx])}")
    avg_values = results[:, :, idx].flatten()
    plt.scatter(
        PA + 0.01 * idx,
        PB + 0.01 * idx,
        s=avg_values * 40,
        c=color,
        alpha=0.3,
    )

plt.xlabel("E(P(A))")
plt.ylabel("E(P(B))")
plt.title("Comparison of Bandit Algorithms")
plt.legend(
    {algo: color for algo, color in zip(algorithms, colors)},
    loc="center left",
    fontsize=15,
    markerscale=0.3,
    bbox_to_anchor=(1, 0.5),
)
plt.grid(True)
plt.savefig("bandit_comparison.png")
