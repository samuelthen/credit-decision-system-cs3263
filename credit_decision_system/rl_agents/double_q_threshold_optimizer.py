import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class DoubleQThresholdOptimizer:
    """Double Q-learning agent that learns optimal decision thresholds based on economic context"""

    def __init__(self, contexts=None, thresholds=None, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.contexts = contexts or ["bleak", "neutral", "positive"]
        self.thresholds = thresholds or np.round(np.linspace(0.5, 0.9, 9), 2).tolist()
        self.context_to_idx = {ctx: i for i, ctx in enumerate(self.contexts)}
        self.actions = list(range(len(self.thresholds)))

        # Two Q-tables for Double Q-learning
        self.Q1 = defaultdict(float)
        self.Q2 = defaultdict(float)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.best_thresholds = {}
        self.rewards_history = []

    def select_action(self, context, explore=True):
        ctx_idx = self.context_to_idx.get(context, 0)
        state = (ctx_idx,)

        if explore and np.random.rand() < self.epsilon:
            action_idx = np.random.choice(self.actions)
        else:
            q_values = [self.Q1[(state, a)] + self.Q2[(state, a)] for a in self.actions]
            action_idx = int(np.argmax(q_values))

        return self.thresholds[action_idx], action_idx

    def update_q_value(self, context, action_idx, reward):
        ctx_idx = self.context_to_idx.get(context, 0)
        state = (ctx_idx,)

        if np.random.rand() < 0.5:
            a_star = np.argmax([self.Q1[(state, a)] for a in self.actions])
            target = reward + self.gamma * self.Q2[(state, a_star)]
            self.Q1[(state, action_idx)] += self.alpha * (target - self.Q1[(state, action_idx)])
        else:
            a_star = np.argmax([self.Q2[(state, a)] for a in self.actions])
            target = reward + self.gamma * self.Q1[(state, a_star)]
            self.Q2[(state, action_idx)] += self.alpha * (target - self.Q2[(state, action_idx)])

    def train(self, X, y_true, risk_scores, economic_context, n_episodes=1000):
        print(f"Training Double Q-learning agent for '{economic_context}' context...")
        ctx_idx = self.context_to_idx.get(economic_context, 0)
        state = (ctx_idx,)

        for _ in range(n_episodes):
            threshold, action_idx = self.select_action(economic_context, explore=True)
            y_pred = (risk_scores >= threshold).astype(int)
            reward = self.calculate_reward(y_true, y_pred, X)

            self.update_q_value(economic_context, action_idx, reward)
            self.rewards_history.append(reward)

        best_action = np.argmax([self.Q1[(state, a)] + self.Q2[(state, a)] for a in self.actions])
        best_threshold = self.thresholds[best_action]

        self.best_thresholds[economic_context] = {
            "threshold": best_threshold,
            "q_value": self.Q1[(state, best_action)] + self.Q2[(state, best_action)]
        }

        print(f"✅ Double Q-learning training complete. Best threshold for '{economic_context}': {best_threshold:.2f}")
        return best_threshold

    def get_optimal_threshold(self, context):
        if context in self.best_thresholds:
            return self.best_thresholds[context]["threshold"]

        fallback = {"bleak": 0.75, "neutral": 0.65, "positive": 0.55}
        return fallback.get(context, 0.65)

    def calculate_reward(self, y_true, y_pred, X):
        reward = 0

        for i in range(len(y_pred)):
            actual = y_true.iloc[i] if hasattr(y_true, 'iloc') else y_true[i]
            pred = y_pred[i]

            credit_amount = X.iloc[i]['credit_amount'] if 'credit_amount' in X.columns else 5000
            duration = X.iloc[i]['duration_months'] if 'duration_months' in X.columns else 24

            interest_rate = 0.08
            interest = credit_amount * interest_rate * (duration / 12)

            if actual == 1 and pred == 0:
                reward -= credit_amount * 0.7
            elif actual == 0 and pred == 1:
                reward -= interest
            elif actual == 0 and pred == 0:
                reward += interest

        return reward

    def visualize_training(self):
        """Visualize reward curve and learned thresholds"""
        plt.figure(figsize=(12, 5))

        # Plot reward history
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards_history)
        plt.title("Rewards During Training")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)

        # Plot best thresholds
        plt.subplot(1, 2, 2)
        contexts = list(self.best_thresholds.keys())
        thresholds = [self.best_thresholds[ctx]['threshold'] for ctx in contexts]
        plt.bar(contexts, thresholds)
        plt.title("Learned Optimal Thresholds by Context")
        plt.xlabel("Economic Context")
        plt.ylabel("Decision Threshold")
        plt.ylim(0.4, 0.9)

        for i, v in enumerate(thresholds):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

        plt.tight_layout()
        return plt
