from .rag.contextual_rag import ContextualRAG
from .rl_agents.rl_threshold_optimizer import RLThresholdOptimizer
from .rl_agents.double_q_threshold_optimizer import DoubleQThresholdOptimizer
from .report.llm_explainer import LLMExplainer
from .data.synthetic_applicant_generator import SyntheticApplicantGenerator
from .data.load_data import load_german_credit_data, prepare_data_splits
from .base_models.base_model_builder import build_standard_models
from .base_models.base_model_tuner import base_model_tuner
from .base_models.fairness_evaluation import visualize_fairness_metrics
from torch.utils.data import DataLoader
from credit_decision_system.base_models.adversarial import (
        extract_sensitive_attributes,
        CreditDataset,
        MainModel,
        Adversary,
        train_adversarial
        )
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime

class CreditDecisionSystem:
    """Main credit decisioning system combining all components"""
    
    def __init__(self, use_llm=True, NLP_AVAILABLE=True, use_adversarial=True):
        """Initialize the credit decision system"""
        self.use_llm = use_llm and NLP_AVAILABLE
        self.use_adversarial = use_adversarial
        
        # Initialize components
        self.rag = ContextualRAG(use_llm=use_llm, client=None, NLP_AVAILABLE=NLP_AVAILABLE, debug=False) # only had use llm at the start, no debug and other info
        self.rl_optimizer = RLThresholdOptimizer()
        self.double_q_optimizer = DoubleQThresholdOptimizer()
        self.explainer = LLMExplainer(use_llm=use_llm)
        self.generator = SyntheticApplicantGenerator()
        
        # Initialize models and data
        self.models = {}
        self.adversary_model = None
        self.preprocessor = None
        self.protected_attributes = []

        
        # Metrics storage
        self.evaluation_results = {}
        self.fairness_metrics = {}
        self.decisions_log = []
    
    def load_data(self):
        """Load and prepare data"""
        # Load German Credit dataset
        X, y, protected_attributes = load_german_credit_data()
        X_train, X_test, y_train, y_test, categorical_cols, numerical_cols = prepare_data_splits(X, y)
        
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.protected_attributes = protected_attributes
        
        # Build standard models
        models, preprocessor = build_standard_models(categorical_cols, numerical_cols)
        self.models = models
        self.preprocessor = preprocessor
        
    def tune_models(self, scoring="roc_auc", cv=5, n_iter=15, n_jobs=-1):
        """
        Tune all models using a two-stage search (randomized → grid), updates self.models.
        """
        print("\n🎯 Tuning all models using RandomizedSearchCV + GridSearchCV")
        tuned_models, best_params = base_model_tuner(
            models=self.models,
            X_train=self.X_train,
            y_train=self.y_train,
            scoring=scoring,
            cv=cv,
            n_iter=n_iter,
            n_jobs=n_jobs
        )
        self.models = tuned_models
        self.tuned_model_params = best_params
        return tuned_models, best_params


    def train_models(self):
        """Train all models"""
        print("Training standard models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)

        
        print("✅ All models trained successfully")
    
    def evaluate_models(self):
        """Evaluate all models and generate evaluation metrics"""
        results = {}
        
        print("\nEvaluating models...")
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            class_report = classification_report(self.y_test, y_pred, output_dict=True)
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(self.y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            # Store results
            results[name] = {
                'confusion_matrix': conf_matrix,
                'classification_report': class_report,
                'roc_auc': roc_auc,
                'fpr': fpr,
                'tpr': tpr,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            # Print basic metrics
            print(f"Accuracy: {class_report['accuracy']:.4f}")
            print(f"Precision (Class 1): {class_report['1']['precision']:.4f}")
            print(f"Recall (Class 1): {class_report['1']['recall']:.4f}")
            print(f"F1 Score (Class 1): {class_report['1']['f1-score']:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
        
        self.evaluation_results = results
        return results
    
    def evaluate_fairness(self, model_name='calibrated_logistic_regression'):
        """Evaluate fairness metrics across protected attributes"""
        fairness_metrics = {}

        if model_name not in self.models:
            print(f"⚠️ Model {model_name} not found.")
            return fairness_metrics

        model = self.models[model_name]
        y_proba = model.predict_proba(self.X_test)[:, 1]

        if 'age' in self.protected_attributes and 'age' in self.X_test.columns:
            self.X_test = self.X_test.copy()
            self.X_test['age_group'] = pd.cut(
                self.X_test['age'],
                bins=[0, 30, 45, 60, 100],
                labels=["young", "adult", "middle_aged", "senior"]
            )
            self.protected_attributes = [
                'age_group' if attr == 'age' else attr
                for attr in self.protected_attributes
            ]

        for attr in self.protected_attributes:
            if attr not in self.X_test.columns:
                continue

            print(f"\n📊 Fairness Evaluation for Protected Attribute: {attr}")

            # Plot and collect metrics
            plot, metrics = visualize_fairness_metrics(
                model, self.X_test, self.y_test, attr,
            )

            # Show the plot live (optional)
            plt.tight_layout()
            plt.show()

            # Print out the metric values
            for metric_set, title, _ in metrics:
                print(f"\n📌 {title}")
                for group, value in metric_set:
                    print(f" - {group}: {value:.3f}")

            fairness_metrics[attr] = metrics

        self.fairness_metrics[model_name] = fairness_metrics
        return fairness_metrics
    
    def train_adversarial_debiasing(self, num_epochs=20, batch_size=64):
        """Train an adversarial debiasing model using gender or foreign_worker as sensitive attributes."""
        print("\n🛡️ Training Adversarial Debiasing Model...")

        # Step 1: Extract sensitive attributes
        X_train_sensitive = extract_sensitive_attributes(self.X_train, self.X)
        X_test_sensitive = extract_sensitive_attributes(self.X_test, self.X)

        sensitive_attr_train = X_train_sensitive["is_female"]  # Or "is_foreign"
        sensitive_attr_test = X_test_sensitive["is_female"]

        # Step 2: Encode features
        X_train_encoded = self.preprocessor.fit_transform(self.X_train)
        X_test_encoded = self.preprocessor.transform(self.X_test)

        # Step 3: Wrap in datasets
        train_dataset = CreditDataset(X_train_encoded, self.y_train, sensitive_attr_train)
        test_dataset = CreditDataset(X_test_encoded, self.y_test, sensitive_attr_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Step 4: Init models
        input_dim = X_train_encoded.shape[1]
        main_model = MainModel(input_dim=input_dim)
        adversary_model = Adversary(input_dim=64)

        # Step 5: Train
        train_adversarial(main_model, adversary_model, train_loader, num_epochs=num_epochs)

        # Save for evaluation later
        self.debiasing_model = main_model
        self.adversary = adversary_model
        self.test_loader_debiasing = test_loader

    
    def prepare_rl_agent(self):
        """Prepare the RL agent with synthetic data for all contexts"""
        print("\nPreparing RL agent with synthetic data...")
        
        # Generate synthetic data for all contexts
        contexts = ["bleak", "neutral", "positive"]
        
        # Use calibrated logistic regression model for predictions
        model_name = 'calibrated_logistic_regression'
        model = self.models.get(model_name)
        
        if not model:
            print("⚠️ Calibrated logistic regression model not found, using first available model")
            model_name = list(self.models.keys())[0]
            model = self.models[model_name]
        
        # Train RL agent for each context
        for context in contexts:
            print(f"\nTraining RL agent for {context} context...")
            
            # Generate synthetic data for this context
            n_samples = 1000
            synthetic_data = self.generator.generate_applicants(n_samples, context)
            
            # Extract features and labels
            X_synthetic = synthetic_data.drop(columns=['risk_score', 'credit_risk', 'economic_context'])
            y_synthetic = synthetic_data['credit_risk']
            
            # Get risk scores from model
            risk_scores = model.predict_proba(X_synthetic)[:, 1]
            
            # Train RL agent
            best_threshold = self.rl_optimizer.train(
                X_synthetic, y_synthetic, risk_scores, context, n_episodes=500)
            
            print(f"✅ Optimal threshold for {context} context: {best_threshold:.2f}")
        
        # Visualize training results
        self.rl_optimizer.visualize_training()
        plt.show()
    
    def make_decision(self, applicant, economic_context_query=None):
        """
        Make a credit decision for an applicant
        
        Args:
            applicant: Applicant features DataFrame (single row)
            economic_context_query: Query string for economic context
        
        Returns:
            Decision dict with all information
        """
        # 1. Determine economic context
        if economic_context_query:
            context_info = self.rag.get_context(economic_context_query)
            economic_context = context_info['classification']
        else:
            # Default context if no query provided
            economic_context = "neutral"
            context_info = {
                "classification": economic_context,
                "reasoning": "Default context (no query provided)",
                "timestamp": datetime.now().isoformat()
            }
        
        # 2. Calculate risk score
        model_name = 'calibrated_logistic_regression'

        risk_score = float(self.models[model_name].predict_proba(pd.DataFrame([applicant]))[0, 1])
        model_used = model_name
        
        # 3. Get optimal threshold based on context
        threshold = self.rl_optimizer.get_optimal_threshold(economic_context)
        
        # 4. Make decision
        # Higher risk score = higher risk of default
        # Reject if risk score >= threshold
        decision = risk_score >= threshold  # True = reject, False = approve
        
        # 5. Generate explanation
        # explanation = self.explainer.generate_explanation(
            # applicant, risk_score, threshold, economic_context, decision)
        explanation = self.explainer.generate_explanation(
            applicant.to_dict(), risk_score, threshold, economic_context, decision)

        
        # 6. Create decision record
        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "applicant": applicant.to_dict(),
            "risk_score": risk_score,
            "model_used": model_used,
            "economic_context": economic_context,
            "context_reasoning": context_info['reasoning'],
            "threshold": threshold,
            "decision": "REJECTED" if decision else "APPROVED",
            "explanation": explanation
        }
        
        # 7. Log the decision
        self.decisions_log.append(decision_record)
        
        return decision_record
    
    def visualize_model_performance(self):
        """Visualize model performance with ROC curves"""
        plt.figure(figsize=(10, 8))
        
        for name, results in self.evaluation_results.items():
            plt.plot(
                results['fpr'], 
                results['tpr'], 
                label=f"{name} (AUC = {results['roc_auc']:.3f})"
            )
        
        plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves for Different Models")
        plt.legend(loc="lower right")
        plt.grid(True)
        
        plt.show()
        
    

    def explain_model_behaviors(self, top_n=20):
        """Explain model behavior: top predictors for LR and one tree for RF"""
        print("\n🔎 Interpreting model behavior...")

        # --- Logistic Regression Coefficients ---
        if "logistic_regression" in self.models:
            logreg_model = self.models["logistic_regression"]
            # Extract named steps from pipeline
            clf = logreg_model.named_steps["classifier"]
            # feature_names = self.preprocessor.named_transformers_["cat"].get_feature_names_out(self.categorical_cols).tolist()
            # all_features = self.numerical_cols + feature_names
            preprocessor = logreg_model.named_steps["preprocess"]
            cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(self.categorical_cols).tolist()
            all_features = self.numerical_cols + cat_features

            print("\n📊 Top Predictors in Logistic Regression:")
            coef_df = pd.DataFrame({
                "feature": all_features,
                "coefficient": clf.coef_[0]
            }).sort_values(by="coefficient", key=abs, ascending=False)

            print(coef_df.head(top_n).to_string(index=False))

            # Optional plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x="coefficient", y="feature", data=coef_df.head(top_n))
            plt.title("Top Predictors in Logistic Regression")
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ Logistic regression model not found.")

        # --- Random Forest Tree Visualization ---
        if "random_forest" in self.models:
            rf_model = self.models["random_forest"]
            clf = rf_model.named_steps["classifier"]

            from sklearn.tree import plot_tree

            plt.figure(figsize=(20, 10))
            plot_tree(
                clf.estimators_[0],
                feature_names=self.preprocessor.get_feature_names_out(),
                filled=True,
                rounded=True,
                max_depth=3,
                fontsize=9
            )
            plt.title("Random Forest - Tree 0 (max depth=3)")
            plt.show()
        else:
            print("⚠️ Random forest model not found.")
    
    def visualize_context_thresholds(self):
        """Visualize optimal thresholds by economic context"""
        contexts = ["bleak", "neutral", "positive"]
        thresholds = [self.rl_optimizer.get_optimal_threshold(ctx) for ctx in contexts]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(contexts, thresholds, color=['#FF6B6B', '#4ECDC4', '#56B870'])
        
        plt.title("Optimal Decision Thresholds by Economic Context")
        plt.xlabel("Economic Context")
        plt.ylabel("Risk Score Threshold")
        plt.ylim(0.5, 0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        
        

    def simulate_thresholds_over_time(self, total_steps=300, smoothing_alpha=0.1):
        """RL threshold adaptation vs fixed threshold (0.5) with full reward tracking across an economic cycle"""
        print("\n📈 Simulating RL threshold adaptation across changing economic conditions...")

        def get_context(step):
            cycle = total_steps // 4
            if step < cycle:
                return "bleak"
            elif step < cycle * 2:
                return "neutral"
            elif step < cycle * 3:
                return "positive"
            else:
                return "bleak"

        def run_simulation(agent, agent_label):
            time_points = []
            thresholds = []
            rewards_over_time = []
            cumulative_rewards_over_time = []
            smoothed_threshold = None
            cumulative_reward = 0

            for step in range(total_steps):
                context = get_context(step)
                synthetic_applicant = self.generator.generate_applicants(1, context)
                X_applicant = synthetic_applicant.drop(columns=['risk_score', 'credit_risk', 'economic_context'])
                y_applicant = synthetic_applicant['credit_risk']
                risk_score = self.models['calibrated_logistic_regression'].predict_proba(X_applicant)[:, 1]

                chosen_threshold, action_idx = agent.select_action(context)
                y_pred = (risk_score >= chosen_threshold).astype(int)
                reward = agent.calculate_reward(y_applicant, y_pred, X_applicant)
                agent.update_q_value(context, action_idx, reward)

                rewards_over_time.append(reward)
                cumulative_reward += reward
                cumulative_rewards_over_time.append(cumulative_reward)

                if hasattr(agent, "Q"):
                    context_idx = agent.context_to_idx[context]
                    best_idx = np.argmax(agent.Q[context_idx])
                else:
                    state = (agent.context_to_idx[context],)
                    q_values = [agent.Q1[(state, a)] + agent.Q2[(state, a)] for a in agent.actions]
                    best_idx = np.argmax(q_values)

                best_threshold = agent.thresholds[best_idx]

                if smoothed_threshold is None:
                    smoothed_threshold = best_threshold
                else:
                    smoothed_threshold = (
                        smoothing_alpha * best_threshold + (1 - smoothing_alpha) * smoothed_threshold
                    )

                time_points.append(step + 1)
                thresholds.append(smoothed_threshold)

            return time_points, thresholds, rewards_over_time, cumulative_rewards_over_time

        def run_fixed_threshold():
            fixed_threshold = 0.5
            rewards = []
            cumulative_rewards = []
            cumulative = 0
            for step in range(total_steps):
                context = get_context(step)
                synthetic_applicant = self.generator.generate_applicants(1, context)
                X_applicant = synthetic_applicant.drop(columns=['risk_score', 'credit_risk', 'economic_context'])
                y_applicant = synthetic_applicant['credit_risk']
                risk_score = self.models['calibrated_logistic_regression'].predict_proba(X_applicant)[:, 1]
                y_pred = (risk_score >= fixed_threshold).astype(int)
                reward = self.rl_optimizer.calculate_reward(y_applicant, y_pred, X_applicant)
                cumulative += reward
                rewards.append(reward)
                cumulative_rewards.append(cumulative)
            return rewards, cumulative_rewards

        # Run simulations
        steps, single_thresholds, single_rewards, single_cumulative = run_simulation(self.rl_optimizer, "Single Q")
        _, double_thresholds, double_rewards, double_cumulative = run_simulation(self.double_q_optimizer, "Double Q")
        fixed_rewards, fixed_cumulative = run_fixed_threshold()

        # Plot 1: Threshold Adaptation
        plt.figure(figsize=(14, 5))
        plt.plot(steps, single_thresholds, label="Single Q Threshold")
        plt.plot(steps, double_thresholds, label="Double Q Threshold")
        cycle = total_steps // 4
        plt.axvline(x=cycle, color='gray', linestyle='--', label="→ Neutral")
        plt.axvline(x=cycle * 2, color='green', linestyle='--', label="→ Positive")
        plt.axvline(x=cycle * 3, color='red', linestyle='--', label="→ Back to Bleak")
        plt.title("Threshold Adaptation Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Decision Threshold")
        plt.ylim(0.4, 0.9)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot 2: Cumulative Reward Comparison
        plt.figure(figsize=(14, 4))
        plt.plot(steps, single_cumulative, label="Single Q Cumulative Reward")
        plt.plot(steps, double_cumulative, label="Double Q Cumulative Reward")
        plt.plot(steps, fixed_cumulative, label="Fixed Threshold (0.5)", linestyle='--')
        plt.title("Cumulative Financial Reward Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Cumulative Reward")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Print Summary
        print(f"\n✅ Simulation complete.")
        print(f"📈 RL Final Cumulative Reward (Single Q): {single_cumulative[-1]:.2f}")
        print(f"📈 RL Final Cumulative Reward (Double Q): {double_cumulative[-1]:.2f}")
        print(f"📉 Fixed Threshold (0.5) Reward: {fixed_cumulative[-1]:.2f}")
    
    
    def tune_all_agents(self, context="neutral", n_trials=10, search_type="random", n_samples=2000):
        """
        Tune hyperparameters for both Q-learning and Double Q-learning agents in a given context
        """
        print(f"\n🎯 Tuning agents for context: {context}")

        # Generate synthetic data
        synthetic_data = self.generator.generate_applicants(n_samples, context)
        X = synthetic_data.drop(columns=["credit_risk", "risk_score", "economic_context"])
        y = synthetic_data["credit_risk"]
        risk_scores = self.models["calibrated_logistic_regression"].predict_proba(X)[:, 1]

        # Tune Q-learning
        q_params, q_reward = self.tune_rl_hyperparameters(
            agent_class=RLThresholdOptimizer,
            context=context,
            X=X,
            y_true=y,
            risk_scores=risk_scores,
            reward_fn=self.rl_optimizer.calculate_reward,
            search_type=search_type,
            n_trials=n_trials
        )

        # Tune Double Q-learning
        dq_params, dq_reward = self.tune_rl_hyperparameters(
            agent_class=DoubleQThresholdOptimizer,
            context=context,
            X=X,
            y_true=y,
            risk_scores=risk_scores,
            reward_fn=self.rl_optimizer.calculate_reward,
            search_type=search_type,
            n_trials=n_trials
        )

        return {
            "context": context,
            "q_learning": {"best_params": q_params, "reward": q_reward},
            "double_q_learning": {"best_params": dq_params, "reward": dq_reward}
        }
        
    def tune_rl_hyperparameters(self, agent_class, context, X, y_true, risk_scores, reward_fn, 
                                search_type="grid", n_trials=20):
        """
        Tune RL agent hyperparameters using training episode reward.
        """
        from itertools import product

        alpha_vals = [0.05, 0.1, 0.2]
        gamma_vals = [0.8, 0.9, 0.95]
        epsilon_vals = [0.1, 0.2, 0.3]
        search_space = list(product(alpha_vals, gamma_vals, epsilon_vals))

        if search_type == "random":
            np.random.shuffle(search_space)
            search_space = search_space[:n_trials]

        best_reward = float('-inf')
        best_params = None

        print(f"🔍 Tuning {agent_class.__name__} hyperparameters for context: {context}")
        
        for alpha, gamma, epsilon in search_space:
            agent = agent_class(alpha=alpha, gamma=gamma, epsilon=epsilon)
            agent.train(X, y_true, risk_scores, economic_context=context, n_episodes=300, reward_fn=reward_fn)
            total_reward = sum(agent.rewards_history)

            if total_reward > best_reward:
                best_reward = total_reward
                best_params = {'alpha': alpha, 'gamma': gamma, 'epsilon': epsilon}
                print(f"✅ New best reward: {best_reward:.2f} with params: {best_params}")
        
        print(f"\n🏁 Best hyperparameters: {best_params}, Total Reward: {best_reward:.2f}")
        return best_params, best_reward
    def tune_and_compare_agents_via_simulation(self, n_trials=10, search_type="random", total_steps=300):
        """
        Tune both Q-learning and Double Q-learning agents using simulation-based reward,
        and plot comparison against fixed thresholds (0.3, 0.5, 0.7).
        """
        from itertools import product

        print("\n🔍 Starting joint tuning of Single Q and Double Q agents...\n")

        alpha_vals = [0.05, 0.1, 0.2]
        gamma_vals = [0.8, 0.9, 0.95]
        epsilon_vals = [0.1, 0.2, 0.3]
        search_space = list(product(alpha_vals, gamma_vals, epsilon_vals))

        if search_type == "random":
            np.random.shuffle(search_space)
            search_space = search_space[:n_trials]

        def simulate_agent(agent, label):
            cycle = total_steps // 4
            smoothed_threshold = None
            cumulative_reward = 0
            thresholds = []
            rewards_over_time = []
            cumulative_rewards = []
            steps = []

            for step in range(total_steps):
                if step < cycle:
                    context = "bleak"
                elif step < cycle * 2:
                    context = "neutral"
                elif step < cycle * 3:
                    context = "positive"
                else:
                    context = "bleak"

                synthetic_applicant = self.generator.generate_applicants(1, context)
                X_app = synthetic_applicant.drop(columns=["risk_score", "credit_risk", "economic_context"])
                y_app = synthetic_applicant["credit_risk"]
                risk_score = self.models["calibrated_logistic_regression"].predict_proba(X_app)[:, 1]

                chosen_threshold, action_idx = agent.select_action(context)
                y_pred = (risk_score >= chosen_threshold).astype(int)
                reward = agent.calculate_reward(y_app, y_pred, X_app)
                agent.update_q_value(context, action_idx, reward)

                cumulative_reward += reward
                rewards_over_time.append(reward)
                cumulative_rewards.append(cumulative_reward)

                # Get current best threshold
                if hasattr(agent, "Q"):
                    context_idx = agent.context_to_idx[context]
                    best_idx = np.argmax(agent.Q[context_idx])
                else:
                    state = (agent.context_to_idx[context],)
                    q_values = [agent.Q1[(state, a)] + agent.Q2[(state, a)] for a in agent.actions]
                    best_idx = np.argmax(q_values)

                best_threshold = agent.thresholds[best_idx]
                if smoothed_threshold is None:
                    smoothed_threshold = best_threshold
                else:
                    smoothed_threshold = 0.1 * best_threshold + 0.9 * smoothed_threshold

                thresholds.append(smoothed_threshold)
                steps.append(step + 1)

            return {
                "label": label,
                "thresholds": thresholds,
                "rewards": rewards_over_time,
                "cumulative_rewards": cumulative_rewards
            }

        def simulate_fixed_threshold(threshold_val):
            rewards = []
            cumulative_rewards = []
            cumulative = 0
            thresholds = []
            steps = []

            for step in range(total_steps):
                cycle = total_steps // 4
                if step < cycle:
                    context = "bleak"
                elif step < cycle * 2:
                    context = "neutral"
                elif step < cycle * 3:
                    context = "positive"
                else:
                    context = "bleak"

                synthetic_applicant = self.generator.generate_applicants(1, context)
                X_app = synthetic_applicant.drop(columns=["risk_score", "credit_risk", "economic_context"])
                y_app = synthetic_applicant["credit_risk"]
                risk_score = self.models["calibrated_logistic_regression"].predict_proba(X_app)[:, 1]
                y_pred = (risk_score >= threshold_val).astype(int)
                reward = self.rl_optimizer.calculate_reward(y_app, y_pred, X_app)
                cumulative += reward
                rewards.append(reward)
                cumulative_rewards.append(cumulative)
                thresholds.append(threshold_val)
                steps.append(step + 1)

            return {
                "label": f"Fixed {threshold_val}",
                "thresholds": thresholds,
                "rewards": rewards,
                "cumulative_rewards": cumulative_rewards
            }

        best_single = {"reward": float("-inf")}
        best_double = {"reward": float("-inf")}

        for alpha, gamma, epsilon in search_space:
            print(f"🎯 Testing α={alpha}, γ={gamma}, ε={epsilon}")

            single_agent = RLThresholdOptimizer()
            single_agent.alpha = alpha
            single_agent.gamma = gamma
            single_agent.epsilon = epsilon

            double_agent = DoubleQThresholdOptimizer(alpha=alpha, gamma=gamma, epsilon=epsilon)

            single_result = simulate_agent(single_agent, "Single Q")
            double_result = simulate_agent(double_agent, "Double Q")

            final_single_reward = single_result["cumulative_rewards"][-1]
            final_double_reward = double_result["cumulative_rewards"][-1]

            print(f" - Single Q Reward: {final_single_reward:.2f}")
            print(f" - Double Q Reward: {final_double_reward:.2f}")

            if final_single_reward > best_single["reward"]:
                best_single.update({
                    "alpha": alpha,
                    "gamma": gamma,
                    "epsilon": epsilon,
                    "reward": final_single_reward,
                    "result": single_result
                })

            if final_double_reward > best_double["reward"]:
                best_double.update({
                    "alpha": alpha,
                    "gamma": gamma,
                    "epsilon": epsilon,
                    "reward": final_double_reward,
                    "result": double_result
                })

        # Simulate fixed thresholds
        fixed_03 = simulate_fixed_threshold(0.3)
        fixed_05 = simulate_fixed_threshold(0.5)
        fixed_07 = simulate_fixed_threshold(0.7)

        print("\n🏁 Best Hyperparameters:")
        print(f"✅ Single Q: α={best_single['alpha']}, γ={best_single['gamma']}, ε={best_single['epsilon']}, Reward={best_single['reward']:.2f}")
        print(f"✅ Double Q: α={best_double['alpha']}, γ={best_double['gamma']}, ε={best_double['epsilon']}, Reward={best_double['reward']:.2f}")

        steps = list(range(1, total_steps + 1))

        # Plot 1: Threshold Adaptation
        plt.figure(figsize=(14, 5))
        plt.plot(steps, best_single['result']['thresholds'], label="Single Q")
        plt.plot(steps, best_double['result']['thresholds'], label="Double Q")
        plt.plot(steps, fixed_03['thresholds'], '--', label="Fixed 0.3")
        plt.plot(steps, fixed_05['thresholds'], '--', label="Fixed 0.5")
        plt.plot(steps, fixed_07['thresholds'], '--', label="Fixed 0.7")
        cycle = total_steps // 4
        plt.axvline(x=cycle, color='gray', linestyle='--', label="→ Neutral")
        plt.axvline(x=cycle * 2, color='green', linestyle='--', label="→ Positive")
        plt.axvline(x=cycle * 3, color='red', linestyle='--', label="→ Back to Bleak")
        plt.title("Threshold Adaptation Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Decision Threshold")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot 2: Cumulative Rewards
        plt.figure(figsize=(14, 4))
        plt.plot(steps, best_single['result']['cumulative_rewards'], label="Single Q")
        plt.plot(steps, best_double['result']['cumulative_rewards'], label="Double Q")
        plt.plot(steps, fixed_03['cumulative_rewards'], '--', label="Fixed 0.3")
        plt.plot(steps, fixed_05['cumulative_rewards'], '--', label="Fixed 0.5")
        plt.plot(steps, fixed_07['cumulative_rewards'], '--', label="Fixed 0.7")
        plt.title("Cumulative Financial Reward Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Cumulative Reward")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return {
            "best_single_q": best_single,
            "best_double_q": best_double,
            "fixed_03": fixed_03,
            "fixed_05": fixed_05,
            "fixed_07": fixed_07
        }
