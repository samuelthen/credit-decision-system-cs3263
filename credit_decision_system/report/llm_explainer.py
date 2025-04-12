import json
from typing import Dict, Optional, Union
from dataclasses import dataclass
import hashlib
from datetime import datetime, timedelta

@dataclass
class ExplanationMetrics:
    """Metrics to track explanation quality and performance"""
    generation_time: float
    explanation_length: int
    confidence_score: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None

class LLMExplainer:
    """
    Generate human-readable explanations for credit decisions.
    
    This class provides both LLM-based and rule-based explanations. The LLM explanation uses an external
    client (for example, an LLM API client) to generate explanations from a natural language prompt.
    When the LLM is not available, a deterministic, rule-based explanation is returned to ensure
    consistent interpretability and fairness.
    """
    
    def __init__(self, use_llm=True, client=None, NLP_AVAILABLE=True, cache_duration_hours=24):
        """
        Initialize the explainer.
        
        Args:
            use_llm (bool): Flag to indicate whether to use an LLM for explanations.
            client: An instance of an LLM API client. Must be provided if use_llm is True.
            NLP_AVAILABLE (bool): Flag to indicate whether necessary NLP libraries are available.
            cache_duration_hours (int): Duration in hours to cache LLM responses.
        """
        self.use_llm = use_llm and NLP_AVAILABLE and client is not None
        self.client = client
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self._response_cache = {}
        self._last_cache_cleanup = datetime.now()
    
    def _validate_inputs(self, applicant_data: Dict, risk_score: float, threshold: float, 
                        context: str, decision: bool) -> None:
        """Validate input parameters for explanation generation"""
        if not isinstance(applicant_data, dict):
            raise ValueError("applicant_data must be a dictionary")
        if not isinstance(risk_score, (int, float)):
            raise ValueError("risk_score must be numeric")
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be numeric")
        if not isinstance(context, str):
            raise ValueError("context must be a string")
        if not isinstance(decision, bool):
            raise ValueError("decision must be a boolean")
        
        # Validate risk score and threshold ranges
        if not 0 <= risk_score <= 1:
            raise ValueError("risk_score must be between 0 and 1")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
    
    def _get_cache_key(self, applicant_data: Dict, risk_score: float, threshold: float, 
                      context: str, decision: bool) -> str:
        """Generate a unique cache key for the input parameters"""
        input_str = json.dumps({
            'applicant_data': applicant_data,
            'risk_score': risk_score,
            'threshold': threshold,
            'context': context,
            'decision': decision
        }, sort_keys=True)
        return hashlib.md5(input_str.encode()).hexdigest()
    
    def _cleanup_cache(self) -> None:
        """Remove expired cache entries"""
        current_time = datetime.now()
        if current_time - self._last_cache_cleanup > timedelta(hours=1):
            self._last_cache_cleanup = current_time
            expired_keys = [
                key for key, (_, timestamp) in self._response_cache.items()
                if current_time - timestamp > self.cache_duration
            ]
            for key in expired_keys:
                del self._response_cache[key]
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Retrieve cached response if available and not expired"""
        self._cleanup_cache()
        if cache_key in self._response_cache:
            response, timestamp = self._response_cache[cache_key]
            if datetime.now() - timestamp <= self.cache_duration:
                return response
        return None
    
    def _calculate_feature_importance(self, applicant_data: Dict) -> Dict[str, float]:
        """Calculate relative importance of different features"""
        # This is a simplified example - in practice, you might use SHAP values or other methods
        importance = {}
        for key, value in applicant_data.items():
            if isinstance(value, (int, float)):
                importance[key] = abs(value) / sum(abs(v) for v in applicant_data.values() 
                                                if isinstance(v, (int, float)))
        return importance
    
    def generate_explanation(self, applicant_data: Dict, risk_score: float, threshold: float, 
                           context: str, decision: bool) -> Dict[str, Union[str, ExplanationMetrics]]:
        """
        Generate an explanation for a credit decision.
        
        Args:
            applicant_data (dict): Applicant features such as 'credit_amount', 'duration_months', etc.
            risk_score (float): The risk score output by the credit model.
            threshold (float): The decision threshold used by the model.
            context (str): Economic context (e.g., "bleak", "neutral", "positive").
            decision (bool): Final decision, where a truthy value typically indicates rejection.
        
        Returns:
            dict: A dictionary containing the explanation and metrics
        """
        start_time = datetime.now()
        
        # Validate inputs
        self._validate_inputs(applicant_data, risk_score, threshold, context, decision)
        
        # Check cache
        cache_key = self._get_cache_key(applicant_data, risk_score, threshold, context, decision)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return {
                'explanation': cached_response,
                'metrics': ExplanationMetrics(
                    generation_time=0.0,
                    explanation_length=len(cached_response),
                    confidence_score=1.0
                )
            }
        
        # If LLM-based explanation is disabled, use the rule-based approach
        if not self.use_llm:
            explanation = self.generate_rule_based_explanation(risk_score, threshold, context, decision)
            metrics = ExplanationMetrics(
                generation_time=(datetime.now() - start_time).total_seconds(),
                explanation_length=len(explanation),
                confidence_score=1.0
            )
            return {'explanation': explanation, 'metrics': metrics}
        
        try:
            # Extract key applicant features
            key_features = {
                'credit_amount': applicant_data.get('credit_amount', 'Unknown'),
                'duration_months': applicant_data.get('duration_months', 'Unknown'),
                'purpose': applicant_data.get('purpose', 'Unknown'),
                'credit_history': applicant_data.get('credit_history', 'Unknown'),
                'savings_account': applicant_data.get('savings_account', 'Unknown'),
                'employment_since': applicant_data.get('employment_since', 'Unknown')
            }
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(applicant_data)
            
            # Construct enhanced prompt
            prompt = f"""
            You are an expert credit decision explainer. Your task is to provide a clear, fair, and 
            transparent explanation for a credit decision. Consider the following details:

            Applicant Information:
            - Loan amount: {key_features['credit_amount']}
            - Duration: {key_features['duration_months']} months
            - Purpose: {key_features['purpose']}
            - Credit history: {key_features['credit_history']}
            - Savings account: {key_features['savings_account']}
            - Employment status: {key_features['employment_since']}

            Decision Details:
            - Risk score: {risk_score:.2f} (higher means higher risk)
            - Decision threshold: {threshold:.2f}
            - Economic context: {context}
            - Final Decision: {"REJECTED" if decision else "APPROVED"}

            Feature Importance:
            {json.dumps(feature_importance, indent=2)}

            Please provide a 3-4 sentence explanation that:
            1. Clearly states the decision
            2. Explains the key factors that influenced the decision
            3. Provides context about the economic conditions
            4. Offers constructive feedback for future applications
            5. Emphasizes fairness and transparency

            Format the explanation in a professional, empathetic tone.
            """
            
            # Generate explanation using LLM
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.7,
                max_tokens=300
            )
            
            explanation = response.choices[0].message.content.strip()
            
            # Cache the response
            self._response_cache[cache_key] = (explanation, datetime.now())
            
            # Calculate metrics
            metrics = ExplanationMetrics(
                generation_time=(datetime.now() - start_time).total_seconds(),
                explanation_length=len(explanation),
                confidence_score=response.choices[0].finish_reason == "stop",
                feature_importance=feature_importance
            )
            
            return {'explanation': explanation, 'metrics': metrics}
            
        except Exception as e:
            print(f"❌ LLM explanation failed: {e}")
            explanation = self.generate_rule_based_explanation(risk_score, threshold, context, decision)
            metrics = ExplanationMetrics(
                generation_time=(datetime.now() - start_time).total_seconds(),
                explanation_length=len(explanation),
                confidence_score=1.0
            )
            return {'explanation': explanation, 'metrics': metrics}
    
    def generate_rule_based_explanation(self, risk_score: float, threshold: float, 
                                      context: str, decision: bool) -> str:
        """
        Generate a rule-based explanation when LLM is unavailable.
        
        Args:
            risk_score (float): The applicant's risk score.
            threshold (float): The decision threshold.
            context (str): Economic context, which may alter the tone of the explanation.
            decision (bool): The final decision; truthy values indicate rejection.
            
        Returns:
            str: A rule-based explanation.
        """
        decision_text = "approved" if not decision else "declined"
        
        context_phrases = {
            "bleak": "During the current economic downturn, we've adjusted our lending criteria to be more conservative.",
            "neutral": "Based on current economic conditions, we're using standard lending criteria.",
            "positive": "In the current favorable economic climate, we are able to offer competitive lending terms."
        }
        
        context_phrase = context_phrases.get(context, "Based on current economic conditions,")
        
        if decision:  # Rejected case
            if risk_score > threshold + 0.1:
                explanation = (
                    f"Your application has been {decision_text}. {context_phrase} "
                    f"Your risk score of {risk_score:.2f} was significantly above our current threshold of {threshold:.2f}, "
                    "indicating a level of risk that exceeds our acceptable limits in this economic climate. "
                    "We encourage you to review your credit profile and consider reapplying in the future."
                )
            else:
                explanation = (
                    f"Your application has been {decision_text}. {context_phrase} "
                    f"Your risk score of {risk_score:.2f} was slightly above our current threshold of {threshold:.2f}. "
                    "We encourage you to strengthen your credit profile and reapply in the future. "
                    "Consider improving your credit score or providing additional collateral."
                )
        else:  # Approved case
            if risk_score < threshold - 0.1:
                explanation = (
                    f"Your application has been {decision_text}! {context_phrase} "
                    f"Your risk score of {risk_score:.2f} was well below our current threshold of {threshold:.2f}, "
                    "reflecting a robust credit profile. We're pleased to offer you competitive terms."
                )
            else:
                explanation = (
                    f"Your application has been {decision_text}. {context_phrase} "
                    f"Your risk score of {risk_score:.2f} was just below our current threshold of {threshold:.2f}. "
                    "We're able to offer you standard terms based on your credit profile."
                )
                
        return explanation