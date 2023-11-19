import random
from collections import defaultdict
import numpy as np


class Blackjack:
    def __init__(self):
        self.deck = [
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "J",
            "Q",
            "K",
            "A",
        ] * 4

        random.shuffle(self.deck)

    def draw(self):
        return self.deck.pop()

    def calculate_score(self, hand):
        score = 0
        ace_count = 0
        for card in hand:
            if card in ["J", "Q", "K"]:
                score += 10
            elif card == "A":
                ace_count += 1
                score += 11
            else:
                score += int(card)
        while score > 21 and ace_count:
            score -= 10
            ace_count -= 1
        return score


class env:
    def __init__(self):
        self.blackjack = Blackjack()
        self.player_hand = [self.blackjack.draw(), self.blackjack.draw()]
        self.dealer_hand = [self.blackjack.draw(), self.blackjack.draw()]

    def get_state(self):
        dealer_first_card = self.blackjack.calculate_score(self.dealer_hand[0])
        player_score = self.blackjack.calculate_score(self.player_hand)
        return (dealer_first_card, player_score)

    def hit(self):
        self.player_hand.append(self.blackjack.draw())
        if self.blackjack.calculate_score(self.player_hand) > 21:
            return -1

    def stand(self):
        while self.blackjack.calculate_score(self.dealer_hand) < 17:
            self.dealer_hand.append(self.blackjack.draw())

    def win_check(self):
        player_score = self.blackjack.calculate_score(self.player_hand)
        dealer_score = self.blackjack.calculate_score(self.dealer_hand)
        if player_score > 21:
            return -1
        elif dealer_score > 21:
            return 1
        elif player_score > dealer_score:
            return 1
        elif player_score < dealer_score:
            return -1
        else:
            return 1


def epsilon_greedy_policy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 1)
    else:
        return np.argmax(Q[state])


def greedy_policy(Q, state):
    return np.argmax(Q[state])


def monte_carlo_control(env, episodes, epsilon):
    each_round = 1000
    win_rate = []
    rewards = []
    Q = defaultdict(lambda: np.zeros(2))
    N = 1
    for i in range(episodes):
        state_action = []
        state = env.get_state()
        while True:
            action = epsilon_greedy_policy(Q, state, epsilon)
            state_action.append((state, action))
            if action == 0:
                flag = env.hit()  # hit
                if flag:
                    break
            else:
                env.stand()
                break

        reward = env.win_check()
        rewards.append(reward)
        for state, action in state_action:
            Q[state][action] = Q[state][action] + (reward - Q[state][action]) / N
        N += 1
        env.__init__()
    for i in range(int(episodes / each_round)):
        win_rate.append(np.sum(rewards[i * each_round : (i + 1) * each_round]))
    return win_rate, Q


def double_Q_learning(env, episodes, alpha=0.1):
    each_round = 1000
    win_rate = []
    rewards = []
    Q1 = defaultdict(lambda: np.zeros(2))
    Q2 = defaultdict(lambda: np.zeros(2))
    for i in range(episodes):
        state_action = []
        state = env.get_state()
        while True:
            if np.sum(Q1[state]) + np.sum(Q2[state]) == 0:
                action = random.randint(0, 1)
            else:
                action = greedy_policy(random.choice([Q1, Q2]), state)
            state_action.append((state, action))
            if action == 0:
                flag = env.hit()  # hit
                if flag:
                    break
            else:
                env.stand()
                break

        reward = env.win_check()
        rewards.append(reward)
        if random.random() < 0.5:
            for state, action in state_action:
                Q1[state][action] = Q1[state][action] + alpha * (
                    reward + Q2[state][greedy_policy(Q1, state)] - Q1[state][action]
                )
        else:
            for state, action in state_action:
                Q2[state][action] = Q2[state][action] + alpha * (
                    reward + Q1[state][greedy_policy(Q2, state)] - Q2[state][action]
                )
        env.__init__()
    for i in range(int(episodes / each_round)):
        win_rate.append(np.sum(rewards[i * each_round : (i + 1) * each_round]))
    return win_rate, Q1, Q2


if __name__ == "__main__":
    env1 = env()
    win_rate, Q = monte_carlo_control(env1, 100000, 0.2)
    print(win_rate)
    # print(Q)
    env2 = env()
    win_rate, Q1, Q2 = double_Q_learning(env2, 100000, 0.2)
    print(win_rate)
    # print(Q1)
    print("done")
