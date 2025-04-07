import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class RLThresholdOptimizer:
    """RL agent that learns optimal decision thresholds based on context"""
    
    def __init__(self, contexts=None, thresholds=None, alpha=0.1, gamma=0.9, epsilon=0.2):
        """
        Initialize the RL agent
        
        Args:
            contexts: List of possible contexts
            thresholds: List of thresholds to explore
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.contexts = contexts or ["bleak", "neutral", "positive"]
        self.thresholds = thresholds or np.round(np.linspace(0.5, 0.9, 9), 2).tolist()
        self.context_to_idx = {ctx: i for i, ctx in enumerate(self.contexts)}
        self.Q = np.zeros((len(self.contexts), len(self.thresholds)))

        # 🧠 Add this to allow hyperparameter tuning
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.rewards_history = []
        self.best_thresholds = {}
    
    def select_action(self, context, explore=True):
        """
        Select threshold action based on context using epsilon-greedy policy
        
        Args:
            context: Economic context (bleak, neutral, positive)
            explore: Whether to use epsilon-greedy exploration
        
        Returns:
            selected threshold value
        """
        ctx_idx = self.context_to_idx.get(context, 0)
        
        # Epsilon-greedy action selection
        if explore and np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, len(self.thresholds))  # Explore
        else:
            action_idx = np.argmax(self.Q[ctx_idx])  # Exploit
        
        return self.thresholds[action_idx], action_idx
    
    def update_q_value(self, context, action_idx, reward):
        """
        Update Q-value based on reward received
        
        Args:
            context: Economic context
            action_idx: Index of selected threshold
            reward: Reward received from environment
        """
        ctx_idx = self.context_to_idx.get(context, 0)
        
        # Q-learning update rule (simplified for stateless case)
        old_value = self.Q[ctx_idx, action_idx]
        self.Q[ctx_idx, action_idx] = old_value + self.alpha * (reward - old_value)
    
    def train(self, X, y_true, risk_scores, economic_context, n_episodes=1000):
        """
        Train the RL agent over multiple episodes
        
        Args:
            X: Feature DataFrame for financial calculations
            y_true: True labels
            risk_scores: Model risk scores (probabilities)
            economic_context: Current economic context
            n_episodes: Number of training episodes
        """
        print(f"Training RL threshold optimizer for '{economic_context}' context...")
        
        ctx_idx = self.context_to_idx.get(economic_context, 0)
        episode_rewards = []
        
        for episode in tqdm(range(n_episodes)):
            # Select threshold using epsilon-greedy
            threshold, action_idx = self.select_action(economic_context)
            
            # Apply threshold to get predictions
            y_pred = (risk_scores >= threshold).astype(int)
            
            # Calculate reward
            reward = self.calculate_reward(y_true, y_pred, X)
            episode_rewards.append(reward)
            
            # Update Q-value
            self.update_q_value(economic_context, action_idx, reward)
        
        # Store best threshold for this context
        best_action_idx = np.argmax(self.Q[ctx_idx])
        best_threshold = self.thresholds[best_action_idx]
        best_q_value = self.Q[ctx_idx, best_action_idx]
        
        self.best_thresholds[economic_context] = {
            'threshold': best_threshold,
            'q_value': best_q_value
        }
        
        self.rewards_history.extend(episode_rewards)
        
        print(f"✅ Training complete. Best threshold for '{economic_context}': {best_threshold:.2f}")
        return best_threshold
    
    def calculate_reward(self, y_true, y_pred, X):
        """
        Calculate reward based on prediction outcomes
        
        Args:
            y_true: True labels (0=good, 1=bad credit)
            y_pred: Predicted labels
            X: Feature DataFrame for financial calculations
        
        Returns:
            Total financial reward
        """
        reward = 0
        
        for i in range(len(y_pred)):
            actual = y_true.iloc[i] if hasattr(y_true, 'iloc') else y_true[i]
            pred = y_pred[i]
            
            # Get financial attributes
            credit_amount = X.iloc[i]['credit_amount'] if 'credit_amount' in X.columns else 5000
            duration = X.iloc[i]['duration_months'] if 'duration_months' in X.columns else 24
            
            # Calculate interest (approximate)
            interest_rate = 0.08  # 8% annual interest
            interest = credit_amount * interest_rate * (duration / 12)
            
            # Financial outcomes:
            if actual == 1 and pred == 0:  # False Negative (bad credit but approved)
                reward -= credit_amount * 0.7  # 70% of loan amount lost
            elif actual == 0 and pred == 1:  # False Positive (good credit but rejected)
                reward -= interest  # Lost interest income
            elif actual == 0 and pred == 0:  # True Negative (good credit, approved)
                reward += interest  # Earned interest
            # True Positive (bad credit, rejected) has zero impact
        
        return reward
    
    def get_optimal_threshold(self, context):
        """Get the optimal threshold for a given context"""
        # If we have learned a threshold for this context, use it
        if context in self.best_thresholds:
            return self.best_thresholds[context]['threshold']
        
        # Otherwise, use a rule-based fallback
        context_to_threshold = {
            "bleak": 0.75,    # Conservative in bad economy
            "neutral": 0.65,  # Moderate in normal economy
            "positive": 0.55  # Permissive in good economy
        }
        
        return context_to_threshold.get(context, 0.65)  # Default to moderate threshold
    
    def visualize_training(self):
        """Visualize training progress and learned thresholds"""
        # Plot rewards history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards_history)
        plt.title("Rewards During Training")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        
        # Plot learned thresholds by context
        plt.subplot(1, 2, 2)
        contexts = list(self.best_thresholds.keys())
        thresholds = [self.best_thresholds[ctx]['threshold'] for ctx in contexts]
        
        plt.bar(contexts, thresholds)
        plt.title("Learned Optimal Thresholds by Context")
        plt.xlabel("Economic Context")
        plt.ylabel("Decision Threshold")
        plt.ylim(0.4, 0.9)
        
        # Add threshold values on bars
        for i, v in enumerate(thresholds):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        return plt