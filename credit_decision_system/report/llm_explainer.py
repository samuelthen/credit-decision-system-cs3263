import json
from typing import Dict, Optional, Union
from dataclasses import dataclass
import hashlib
from datetime import datetime, timedelta

@dataclass
class ExplanationMetrics:
    """Metrics to track explanation quality and performance."""
    generation_time: float
    explanation_length: int
    confidence_score: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None

class LLMExplainer:
    """
    Generate human-readable explanations for credit decisions.
    
    This module provides both LLM-based and rule-based explanations. The LLM explanation leverages an external
    client (e.g., an LLM API client) to generate narratives from carefully crafted prompts. When the LLM is not
    available, a deterministic rule-based explanation is provided. The current implementation integrates techniques 
    inspired by research in "LLMs for Explainable AI: A Comprehensive Survey" (Bilal et al., 2025), including:
      - Post-hoc analysis that differentiates local (specific) and global (general) influences.
      - Intrinsic interpretability using Chain of Thought reasoning to detail step-by-step logic.
      - Human-centered narrative instructions that produce explanations in a third-person perspective.
      
    Additionally, the module can incorporate regression coefficients for feature importance. When provided, these
    coefficients are used to weight applicant data, enhancing the explanation by aligning the reported importance
    with the underlying predictive model's coefficients.
    
    The final explanation is evaluated on clarity, factual accuracy, logical coherence, and robustness.
    """
    
    def __init__(self, use_llm=True, client=None, NLP_AVAILABLE=True, cache_duration_hours=24,
                 regression_coefficients: Optional[Dict[str, float]] = None):
        """
        Initialize the explainer.
        
        Args:
            use_llm (bool): Whether to employ LLM-based explanations.
            client: An instance of an LLM API client.
            NLP_AVAILABLE (bool): Whether necessary NLP libraries are available.
            cache_duration_hours (int): Duration in hours for caching responses.
            regression_coefficients (Optional[Dict[str, float]]): A mapping from feature names to their regression coefficients.
        """
        self.use_llm = use_llm and NLP_AVAILABLE and client is not None
        self.client = client
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self._response_cache = {}
        self._last_cache_cleanup = datetime.now()
        self.coefficients = regression_coefficients  # New optional parameter
    
    def _validate_inputs(self, applicant_data: Dict, risk_score: float, threshold: float, 
                         context: str, decision: bool) -> None:
        """Validate input parameters for explanation generation."""
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
        
        if not 0 <= risk_score <= 1:
            raise ValueError("risk_score must be between 0 and 1")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
    
    def _get_cache_key(self, applicant_data: Dict, risk_score: float, threshold: float, 
                       context: str, decision: bool) -> str:
        """Generate a unique cache key for the input parameters."""
        input_str = json.dumps({
            'applicant_data': applicant_data,
            'risk_score': risk_score,
            'threshold': threshold,
            'context': context,
            'decision': decision
        }, sort_keys=True)
        return hashlib.md5(input_str.encode()).hexdigest()
    
    def _cleanup_cache(self) -> None:
        """Remove expired cache entries."""
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
        """Retrieve cached response if available and not expired."""
        self._cleanup_cache()
        if cache_key in self._response_cache:
            response, timestamp = self._response_cache[cache_key]
            if datetime.now() - timestamp <= self.cache_duration:
                return response
        return None
    
    def _calculate_feature_importance(self, applicant_data: Dict) -> Dict[str, float]:
        """
        Calculate the relative importance of different features.
        
        If regression coefficients are provided, the importance for each feature is calculated as:
          importance[f] = |value * coefficient| / (sum_{f in features} |value * coefficient|)
        Otherwise, a simple normalized absolute value is used.
        """
        importance = {}
        # If regression coefficients are provided, use them.
        if self.coefficients:
            weighted_values = {}
            for key, value in applicant_data.items():
                if isinstance(value, (int, float)) and key in self.coefficients:
                    weighted_values[key] = abs(value * self.coefficients[key])
            total = sum(weighted_values.values())
            if total > 0:
                for key, val in weighted_values.items():
                    importance[key] = val / total
            return importance
        
        # Fallback to the original approach if no coefficients provided.
        numeric_values = [abs(value) for value in applicant_data.values() if isinstance(value, (int, float))]
        if not numeric_values:
            return importance
        total = sum(numeric_values)
        for key, value in applicant_data.items():
            if isinstance(value, (int, float)):
                importance[key] = abs(value) / total
        return importance
    
    def _chain_of_thought_prompt(self) -> str:
        """
        Generate a Chain-of-Thought (CoT) reasoning prompt segment.
        
        Instruct the model to logically outline the sequential steps underlying the decision.
        """
        return (
            "Using a Chain of Thought approach, first analyze the applicant's credit history, then assess how the risk "
            "score compares with the threshold, and finally consider the economic context. Explain each step sequentially."
        )
    
    def _posthoc_insight_prompt(self) -> str:
        """
        Generate a post-hoc explanation prompt segment.
        
        Instruct the model to highlight which features were most influential locally (for this specific decision) and "
        "how these features generally influence similar credit decisions."
        """
        return (
            "Provide post-hoc insights by identifying key features that most influenced this decision and comment on their general impact."
        )
    
    def _human_centered_prompt(self) -> str:
        """
        Generate a human-centered narrative prompt segment.
        
        Instruct the model to write in a third-person narrative (e.g., 'This applicant ...', 'It is observed that ...')
        and offer constructive, counterfactual feedback.
        """
        return (
            "Frame the explanation in third-person narrative. For example, 'This applicant exhibits...' or 'It is observed "
            "that...'. Also, include constructive feedback suggesting how improvements in certain areas might affect the outcome."
        )
    
    def generate_explanation(self, applicant_data: Dict, risk_score: float, threshold: float, 
                             context: str, decision: bool) -> Dict[str, Union[str, ExplanationMetrics]]:
        """
        Generate an explanation for a credit decision.
        
        The explanation is designed to be transparent, fair, and human-readable. It incorporates post-hoc analysis, 
        Chain-of-Thought reasoning, and human-centered narratives in a third-person tone.
        
        Args:
            applicant_data (dict): Applicant features (e.g., 'credit_amount', 'duration_months').
            risk_score (float): The risk score produced by the credit model.
            threshold (float): The decision threshold.
            context (str): Economic context (e.g., "bleak", "neutral", "positive").
            decision (bool): Final decision (True if rejected, False if approved).
        
        Returns:
            dict: Contains the generated explanation and associated metrics.
        """
        start_time = datetime.now()
        
        self._validate_inputs(applicant_data, risk_score, threshold, context, decision)
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
        
        # Use rule-based explanation if LLM is disabled.
        if not self.use_llm:
            explanation = self.generate_rule_based_explanation(risk_score, threshold, context, decision)
            metrics = ExplanationMetrics(
                generation_time=(datetime.now() - start_time).total_seconds(),
                explanation_length=len(explanation),
                confidence_score=1.0
            )
            return {'explanation': explanation, 'metrics': metrics}
        
        try:
            key_features = {
                'credit_amount': applicant_data.get('credit_amount', 'Unknown'),
                'duration_months': applicant_data.get('duration_months', 'Unknown'),
                'purpose': applicant_data.get('purpose', 'Unknown'),
                'credit_history': applicant_data.get('credit_history', 'Unknown'),
                'savings_account': applicant_data.get('savings_account', 'Unknown'),
                'employment_since': applicant_data.get('employment_since', 'Unknown')
            }
            
            feature_importance = self._calculate_feature_importance(applicant_data)
            
            # Construct the enhanced prompt using academic insights and the new techniques.
            prompt = f"""
            You are an expert credit decision explainer. Using a third-person narrative, provide a clear, fair, and transparent 
            explanation for this credit decision.

            Applicant Information:
            - Loan amount: {key_features['credit_amount']}
            - Duration: {key_features['duration_months']} months
            - Purpose: {key_features['purpose']}
            - Credit history: {key_features['credit_history']}
            - Savings account: {key_features['savings_account']}
            - Employment status: {key_features['employment_since']}

            Decision Details:
            - Risk score: {risk_score:.2f} (a higher score indicates higher risk)
            - Decision threshold: {threshold:.2f}
            - Economic context: {context}
            - Final Decision: {"REJECTED" if decision else "APPROVED"}

            Feature Importance:
            {json.dumps(feature_importance, indent=2)}

            {self._chain_of_thought_prompt()}
            {self._posthoc_insight_prompt()}
            {self._human_centered_prompt()}

            Academic Insights:
            Based on "LLMs for Explainable AI: A Comprehensive Survey" (Bilal et al., 2025), the explanation should reflect clarity, 
            factual accuracy, logical coherence, and robustness. It must address both local influences specific to this applicant and 
            broader trends across similar cases.

            Provide a 3-4 sentence explanation in third-person style, such as "This applicant..." and include constructive feedback 
            on how modifications in key areas might affect future decisions.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.7,
                max_tokens=300
            )
            
            explanation = response.choices[0].message.content.strip()
            self._response_cache[cache_key] = (explanation, datetime.now())
            
            metrics = ExplanationMetrics(
                generation_time=(datetime.now() - start_time).total_seconds(),
                explanation_length=len(explanation),
                confidence_score=1.0 if response.choices[0].finish_reason == "stop" else 0.0,
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
        Generate a rule-based explanation when the LLM is unavailable.
        
        The explanation is provided in third-person narrative.
        
        Args:
            risk_score (float): The applicant's risk score.
            threshold (float): The decision threshold.
            context (str): Economic context.
            decision (bool): Final decision (True for rejection, False for approval).
            
        Returns:
            str: The generated explanation.
        """
        decision_text = "approved" if not decision else "declined"
        context_phrases = {
            "bleak": "During the current economic downturn, lending criteria have been tightened.",
            "neutral": "Based on current economic conditions, standard lending practices are followed.",
            "positive": "In a favorable economic climate, competitive lending terms are offered."
        }
        context_phrase = context_phrases.get(context, "Based on current economic conditions,")
        
        if decision:  # Rejected case
            if risk_score > threshold + 0.1:
                explanation = (
                    f"This applicant has been {decision_text} as their risk score of {risk_score:.2f} significantly exceeds the threshold "
                    f"of {threshold:.2f}. {context_phrase} The analysis indicates that the financial profile does not meet the risk criteria, "
                    "suggesting that improvements in key areas are required before reapplication."
                )
            else:
                explanation = (
                    f"This applicant has been {decision_text} since their risk score of {risk_score:.2f} is slightly above the threshold of "
                    f"{threshold:.2f}. {context_phrase} It is recommended that the applicant enhances their financial metrics for a more favorable outcome."
                )
        else:  # Approved case
            if risk_score < threshold - 0.1:
                explanation = (
                    f"This applicant has been {decision_text} as their risk score of {risk_score:.2f} is well below the threshold of {threshold:.2f}. "
                    f"{context_phrase} The financial profile indicates strong creditworthiness, thus justifying the competitive lending terms offered."
                )
            else:
                explanation = (
                    f"This applicant has been {decision_text} with a risk score of {risk_score:.2f} that is near the threshold of {threshold:.2f}. "
                    f"{context_phrase} Although the credit profile meets the basic criteria, there is an opportunity for enhancement to further optimize the decision."
                )
                
        return explanation
