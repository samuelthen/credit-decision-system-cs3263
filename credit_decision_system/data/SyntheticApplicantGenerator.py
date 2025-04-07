import numpy as np
import pandas as pd
import random

class SyntheticApplicantGenerator:
    """Generate synthetic credit applicants with context-dependent risk profiles"""
    
    def __init__(self, seed=42):
        """Initialize the generator"""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Define context-dependent risk profiles
        self.context_risk_profiles = {
            "bleak": {
                "base_risk": 0.6,  # Higher base risk during economic downturn
                "credit_amount_range": (2000, 15000),  # Lower max credit amounts
                "duration_range": (6, 36),  # Shorter loan durations
                "age_range": (25, 65),  # Similar age distribution
                "savings_levels": ["little", "moderate", "little", "little"],  # Lower savings
                "job_types": ["unskilled", "skilled", "skilled", "unemployed"]  # More unemployment
            },
            "neutral": {
                "base_risk": 0.3,  # Moderate base risk
                "credit_amount_range": (2000, 20000),  # Standard credit amounts
                "duration_range": (6, 48),  # Standard durations
                "age_range": (18, 70),  # Standard age range
                "savings_levels": ["little", "little", "moderate", "rich"],  # Mixed savings
                "job_types": ["unskilled", "skilled", "skilled", "management"]  # Mixed employment
            },
            "positive": {
                "base_risk": 0.15,  # Lower base risk in good economy
                "credit_amount_range": (5000, 30000),  # Higher credit amounts
                "duration_range": (12, 60),  # Longer loan durations
                "age_range": (18, 70),  # Standard age range
                "savings_levels": ["little", "moderate", "rich", "rich"],  # Higher savings
                "job_types": ["skilled", "skilled", "management", "management"]  # Better employment
            }
        }
    
    def generate_applicants(self, n, context="neutral"):
        """
        Generate synthetic credit applicants
        
        Args:
            n: Number of applicants to generate
            context: Economic context (bleak, neutral, positive)
        
        Returns:
            DataFrame of synthetic applicants
        """
        # Get profile for this context
        profile = self.context_risk_profiles.get(context, self.context_risk_profiles["neutral"])
        
        applicants = []
        
        for _ in range(n):
            # Basic attributes
            credit_amount = np.random.randint(*profile["credit_amount_range"])
            duration = np.random.randint(*profile["duration_range"])
            age = np.random.randint(*profile["age_range"])
            
            # Generate categorical attributes
            savings = random.choice(profile["savings_levels"])
            job = random.choice(profile["job_types"])
            purpose = random.choice([
                "car", "furniture", "radio/TV", "domestic appliances", 
                "repairs", "education", "business", "vacation"
            ])
            sex = random.choice(["male", "female"])
            
            # Generate credit history
            credit_history = random.choice([
                "no credits", "all paid", "existing paid", 
                "delayed previously", "critical account"
            ])
            
            # Calculate base risk score
            base_risk = profile["base_risk"]
            
            # Adjust risk based on attributes
            risk_adjustments = 0
            
            # Lower risk for: higher age, better job, more savings
            if age > 40:
                risk_adjustments -= 0.05
            if job in ["management", "skilled"]:
                risk_adjustments -= 0.1
            if savings in ["rich", "moderate"]:
                risk_adjustments -= 0.1
            
            # Higher risk for: larger loan amounts, longer duration, certain purposes
            if credit_amount > profile["credit_amount_range"][1] * 0.7:
                risk_adjustments += 0.1
            if duration > profile["duration_range"][1] * 0.7:
                risk_adjustments += 0.05
            if purpose in ["vacation", "business"]:
                risk_adjustments += 0.1
            
            # Credit history has significant impact
            if credit_history in ["critical account", "delayed previously"]:
                risk_adjustments += 0.2
            elif credit_history in ["existing paid", "all paid"]:
                risk_adjustments -= 0.15
            
            # Final risk score
            risk_score = max(0.01, min(0.99, base_risk + risk_adjustments))
            
            # Determine label (1=bad credit risk, 0=good credit risk)
            is_risky = np.random.rand() < risk_score
            label = 1 if is_risky else 0
            
            # Create applicant record
            applicant = {
                "checking_account_status": random.choice(["no account", "<0", "0<=X<200", ">=200"]),
                "duration_months": duration,
                "credit_history": credit_history,
                "purpose": purpose,
                "credit_amount": credit_amount,
                "savings_account": savings,
                "employment_since": random.choice(["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]),
                "installment_rate": random.randint(1, 4),
                "personal_status_sex": sex,
                "age": age,
                "job": job,

                # 🆕 Add all remaining expected columns
                "housing": random.choice(["own", "for free", "rent"]),
                "present_residence": random.randint(1, 4),
                "existing_credits": random.randint(1, 4),
                "num_dependents": random.randint(1, 2),
                "other_debtors": random.choice(["none", "guarantor", "co-applicant"]),
                "telephone": random.choice(["yes", "no"]),
                "property": random.choice(["real estate", "building society savings", "car", "unknown"]),
                "foreign_worker": random.choice(["yes", "no"]),
                "other_installments": random.choice(["none", "bank", "stores"]),

                "risk_score": risk_score,
                "credit_risk": label,
                "economic_context": context
            }
            
            applicants.append(applicant)
        
        return pd.DataFrame(applicants)
    
    def generate_multi_context_dataset(self, n_total=1000, distribution=None):
        """
        Generate a dataset with multiple economic contexts
        
        Args:
            n_total: Total number of applicants
            distribution: Dict of context proportions (default: 50% bleak, 30% neutral, 20% positive)
        
        Returns:
            Combined DataFrame of synthetic applicants
        """
        if distribution is None:
            distribution = {
                "bleak": 0.5,
                "neutral": 0.3,
                "positive": 0.2
            }
        
        # Validate distribution sums to approximately 1
        total = sum(distribution.values())
        if not 0.99 <= total <= 1.01:
            print(f"⚠️ Distribution sums to {total}, not 1.0. Normalizing...")
            distribution = {k: v/total for k, v in distribution.items()}
        
        # Generate applicants for each context
        all_applicants = []
        for context, proportion in distribution.items():
            n_context = int(n_total * proportion)
            context_applicants = self.generate_applicants(n_context, context)
            all_applicants.append(context_applicants)
        
        # Combine and shuffle
        combined_df = pd.concat(all_applicants, ignore_index=True)
        return combined_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
