class LLMExplainer:
    """Generate human-readable explanations for credit decisions"""
    
    def __init__(self, use_llm=True, client=None, NLP_AVAILABLE=True):
        """Initialize the explainer"""
        self.use_llm = use_llm and NLP_AVAILABLE and client is not None
        self.client = client
    
    def generate_explanation(self, applicant_data, risk_score, threshold, context, decision):
        """
        Generate an explanation for a credit decision
        
        Args:
            applicant_data: Applicant features
            risk_score: Model risk score
            threshold: Decision threshold used
            context: Economic context
            decision: Final decision (approved/rejected)
        
        Returns:
            Human-readable explanation
        """
        if not self.use_llm:
            return self.generate_rule_based_explanation(
                risk_score, threshold, context, decision)
        
        try:
            # Extract key applicant features for explanation
            key_features = {
                'credit_amount': applicant_data.get('credit_amount', 'Unknown'),
                'duration_months': applicant_data.get('duration_months', 'Unknown'),
                'purpose': applicant_data.get('purpose', 'Unknown'),
                'credit_history': applicant_data.get('credit_history', 'Unknown'),
                'savings_account': applicant_data.get('savings_account', 'Unknown'),
                'employment_since': applicant_data.get('employment_since', 'Unknown')
            }
            
      # Continue from previous code...
            
            prompt = f"""
            You are a helpful credit decision explainer. Create a clear, concise explanation for a credit decision:
            
            Details:
            - Loan amount: {key_features['credit_amount']}
            - Duration: {key_features['duration_months']} months
            - Purpose: {key_features['purpose']}
            - Credit history: {key_features['credit_history']}
            - Risk score: {risk_score:.2f} (higher means higher risk)
            - Economic context: {context}
            - Decision threshold: {threshold:.2f}
            - Decision: {"REJECTED" if decision else "APPROVED"}
            
            Generate a clear, professional 3-4 sentence explanation that emphasizes fairness and contextual factors.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.7,
                max_tokens=200
            )
            
            explanation = response.choices[0].message.content.strip()
            return explanation
            
        except Exception as e:
            print(f"❌ LLM explanation failed: {e}")
            return self.generate_rule_based_explanation(
                risk_score, threshold, context, decision)
    
    def generate_rule_based_explanation(self, risk_score, threshold, context, decision):
        """Generate a rule-based explanation when LLM is unavailable"""
        decision_text = "approved" if not decision else "declined"
        
        # Context-specific language
        context_phrases = {
            "bleak": "During the current economic downturn, we've adjusted our lending criteria to be more conservative.",
            "neutral": "Based on current economic conditions, we're using standard lending criteria.",
            "positive": "In the current favorable economic climate, we're able to offer competitive lending terms."
        }
        
        context_phrase = context_phrases.get(context, "Based on current economic conditions")
        
        if decision:  # Rejected
            if risk_score > threshold + 0.1:
                explanation = f"Your application has been {decision_text}. {context_phrase} Your risk score of {risk_score:.2f} was significantly above our current threshold of {threshold:.2f}, indicating higher than acceptable risk in this economic climate."
            else:
                explanation = f"Your application has been {decision_text}. {context_phrase} Your risk score of {risk_score:.2f} was slightly above our current threshold of {threshold:.2f}. We encourage you to strengthen your credit profile and apply again in the future."
        else:  # Approved
            if risk_score < threshold - 0.1:
                explanation = f"Your application has been {decision_text}! {context_phrase} Your risk score of {risk_score:.2f} was well below our current threshold of {threshold:.2f}, indicating a strong credit profile."
            else:
                explanation = f"Your application has been {decision_text}. {context_phrase} Your risk score of {risk_score:.2f} was just below our current threshold of {threshold:.2f}."
                
        return explanation