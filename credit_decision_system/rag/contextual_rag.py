from sentence_transformers import SentenceTransformer
from context_awareness_gate import ContextAwarenessGate
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
import json

class ContextualRAG:
    """
    Retrieval-Augmented Generation (RAG) system for economic context classification.

    This implementation integrates the Context Awareness Gate (CAG) using the Vector Candidates (VC) algorithm,
    which allows the system to decide whether retrieval of external economic context is necessary
    before querying a large language model (LLM).

    The system combines:
        - Efficient retrieval from a domain-specific FAISS vector index
        - Rule-based or LLM-based classification logic
        - Dynamic query routing via CAG for optimized performance and relevance
    """
    def __init__(self, use_llm=True, client=None, NLP_AVAILABLE=True):
        """
        Initialize the ContextualRAG pipeline.

        Args:
            use_llm (bool): Whether to enable LLM for reasoning and classification.
            client: LLM API client (e.g., OpenAI's GPT) for inference if enabled.
            NLP_AVAILABLE (bool): Flag to enable fallback if embedding model fails or is unavailable.
        """
        self.use_llm = use_llm and client is not None
        self.client = client
        self.documents = self._load_documents()

        if NLP_AVAILABLE:
            try:
                # Load embedding model
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

                # Prepare document embeddings
                all_docs, all_contexts = [], []
                for context, docs in self.documents.items():
                    all_docs.extend(docs)
                    all_contexts.extend([context] * len(docs))

                self.doc_embeddings = self.embedder.encode(all_docs, convert_to_numpy=True)
                self.contexts = all_contexts

                # Build FAISS index for fast retrieval
                dimension = self.doc_embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(self.doc_embeddings)

                # Initialize Context Awareness Gate
                self.cag = ContextAwarenessGate(
                    embedder=self.embedder,
                    documents=self.documents,
                    threshold=0.35,
                    top_k=3,
                    use_llm_signal=True,
                    llm_client=self.client,
                    debug=self.debug
                )

                print("✅ ContextualRAG initialized with CAG")
            except Exception as e:
                print(f"❌ Error initializing RAG: {e}")
                self.use_llm = False
        else:
            print("⚠️ NLP not available. Falling back to rule-based classification.")

    def _load_documents(self):
        """
        Load predefined macroeconomic context documents grouped by classification label.

        Returns:
            dict: A dictionary where keys are labels (e.g., "bleak", "neutral", "positive")
                  and values are lists of short economic indicators or reports.
        """
        return {
            "bleak": [
                "Singapore's tech industry is facing headwinds with over 5,000 layoffs reported in Q2 alone.",
                "Venture capital funding in Southeast Asia has dropped 40% compared to the previous year.",
                "The demand for IT and software engineers has declined sharply, especially in the fintech sector.",
                "Startups report significant difficulties raising follow-on funding in the current climate.",
                "The NASDAQ tech index has fallen 15% over the last quarter, dragging investor sentiment in the region.",
                "MAS has issued a warning about capital outflows from riskier sectors, including emerging tech.",
                "Many early-stage startups in Singapore are freezing hiring and cutting operational costs.",
                "Several co-working tech hubs have seen declining occupancy due to shutdowns and cost-cutting.",
                "The tech talent market has cooled, with a 30% drop in tech job postings year-on-year.",
                "SMEs in the software sector report delayed payments and tighter credit lines from banks."
            ],
            "neutral": [
                "Singapore's tech industry shows mixed signals with some sectors growing while others consolidate.",
                "Venture capital funding in Southeast Asia remains stable compared to the previous quarter.",
                "The demand for IT and software engineers is holding steady across most sectors.",
                "Startups report moderate success in raising follow-on funding in the current climate.",
                "The NASDAQ tech index has moved sideways over the last quarter, suggesting market indecision.",
                "MAS maintains its current outlook on technology investments with no new restrictions.",
                "Most tech companies in Singapore are maintaining current headcount levels.",
                "Co-working tech hubs report stable occupancy rates with moderate new membership sign-ups.",
                "The tech talent market is balanced, with job postings matching historical averages.",
                "SMEs in the software sector report normal payment cycles and standard credit access."
            ],
            "positive": [
                "Singapore's tech industry is booming with a 25% increase in new company registrations this quarter.",
                "Venture capital funding in Southeast Asia has surged 35% compared to the previous year.",
                "The demand for IT and software engineers has reached an all-time high across all tech sectors.",
                "Startups report unprecedented success in raising funding, with average rounds increasing by 40%.",
                "The NASDAQ tech index has gained 20% over the last quarter, boosting investor confidence.",
                "MAS has introduced new incentives for financial technology innovation and investments.",
                "Tech companies across Singapore are competing aggressively for talent with increased compensation.",
                "Co-working tech hubs are expanding rapidly with waiting lists for premium spaces.",
                "The tech talent market is red hot, with a 45% increase in job postings year-on-year.",
                "SMEs in the software sector report excellent cash flow and easy access to credit and capital."
            ]
        }

    def retrieve_context(self, applicant_profile, top_k=3):
        """
        Retrieve economic and policy context relevant to a specific applicant profile.

        The method constructs a dynamic, context-aware query using applicant attributes
        such as loan type, industry, credit history, and current quarter. This query is
        embedded and used to retrieve the most semantically similar documents via FAISS.

        Args:
            applicant_profile (dict): Structured data containing applicant-specific features. Example:
                {
                    "industry": "hospitality",
                    "loan_type": "SME",
                    "credit_history": "limited",
                    "location": "Singapore",
                    "current_quarter": "Q2 2025"
                }
            top_k (int): Number of top similar documents to retrieve.

        Returns:
            tuple:
                - result_docs (list[str]): List of context-relevant document snippets.
                - majority_context (str): Dominant label among retrieved documents (e.g., 'bleak', 'positive').
                - query (str): The dynamically generated context-aware query string.
        """
        # Construct context-aware query using applicant profile
        query = (
            f"macroeconomic and regulatory impact of {applicant_profile.get('loan_type', '')} "
            f"loans in {applicant_profile.get('current_quarter', '')}, "
            f"{applicant_profile.get('industry', '')} sector, "
            f"with {applicant_profile.get('credit_history', '')} credit history"
        )

        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)

        # Determine which context label is most represented among the top documents
        retrieved_contexts = [self.contexts[i] for i in indices[0]]
        context_counts = {ctx: retrieved_contexts.count(ctx) for ctx in set(retrieved_contexts)}
        majority_context = max(context_counts, key=context_counts.get)

        # Return documents from the winning context label
        result_docs = [self.documents[majority_context][i % len(self.documents[majority_context])]
                       for i, ctx in enumerate(retrieved_contexts) if ctx == majority_context]

        return result_docs, majority_context, query

    def classify_context_with_llm(self, applicant_profile):
        """
        Generate a risk classification using LLM, based on applicant profile + retrieved context.

        This method builds a tailored prompt combining macroeconomic documents and the
        applicant’s structured profile. The prompt is sent to an LLM, which returns a
        risk classification, rationale, and optional regulatory note.

        Args:
            applicant_profile (dict): Structured profile containing features relevant to credit decisioning.

        Returns:
            dict: {
                "risk_classification": "low" | "medium" | "high",
                "reasoning": "...",
            }
        """
        if not self.use_llm:
            return self.classify_context_rule_based(query)

        try:
            retrieved_docs, majority_context, personalized_query = self.retrieve_context(applicant_profile)
            context = "\n".join(retrieved_docs)

            # Generate a context-aware, applicant-specific prompt
            prompt = f"""
            You are a credit policy AI assistant.

            Based on the following applicant profile and macroeconomic context, classify the risk outlook:
            Profile: {json.dumps(applicant_profile, indent=2)}
            
            Context:
            {context}

            Provide your answer in this JSON format:
            {{
                "risk_classification": "low | medium | high",
                "reasoning": "short justification",
                "policy_consideration": "optional regulatory note if applicable"
            }}
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.3,
                max_tokens=300
            )

            return json.loads(response.choices[0].message.content.strip())

        except Exception as e:
            print(f"❌ LLM classification failed: {e}")
            return self.classify_context_rule_based(query)

    def classify_context_rule_based(self, applicant_profile):
        """
        Fallback risk classifier using keyword-based heuristics on applicant profile.

        Used when:
        - LLM is unavailable
        - Retrieval is skipped by the Context Awareness Gate (CAG)

        This performs a basic scan for economic sentiment keywords and applies a scoring heuristic.

        Args:
            applicant_profile (dict): Structured dictionary of applicant features.

        Returns:
            dict: {
                "risk_classification": "low" | "medium" | "high",
                "reasoning": "...",
                "policy_consideration": "..." (optional)
            }
        """
        query = json.dumps(applicant_profile).lower()
        bleak_keywords = ["recession", "inflation", "layoffs", "decline"]
        positive_keywords = ["growth", "incentive", "relief", "expansion"]

        query_lower = query.lower()
        bleak_score = sum(word in query_lower for word in bleak_keywords)
        positive_score = sum(word in query_lower for word in positive_keywords)

        if bleak_score > positive_score:
            return {
                "risk_classification": "high",
                "reasoning": "Negative indicators found in applicant context.",
                "policy_consideration": "Recommend conservative lending approach."
            }
        elif positive_score > bleak_score:
            return {
                "risk_classification": "low",
                "reasoning": "Positive market signals for applicant profile.",
                "policy_consideration": ""
            }
        else:
            return {
                "risk_classification": "medium",
                "reasoning": "Mixed or unclear signals in profile.",
                "policy_consideration": ""
            }

    def get_applicant_context(self, applicant_profile):
        """
        Main method for context-aware credit assessment using the applicant profile.

        This function intelligently routes the decision flow:
        1. Uses the Context Awareness Gate (CAG) to decide if retrieval is needed.
        2. If retrieval is unnecessary, uses a rule-based fallback.
        3. If retrieval is needed, invoke prompt-driven LLM classification.

        Args:
            applicant_profile (dict): Structured profile with applicant features such as:
                - loan_type
                - industry
                - credit_history
                - location
                - current_quarter

        Returns:
            dict: {
                "risk_classification": "low" | "medium" | "high",
                "reasoning": "...",
                "policy_consideration": "...",
                "timestamp": datetime string
            }
        """
        query_str = f"{applicant_profile.get('loan_type', '')} in {applicant_profile.get('industry', '')}"

        # Check with Context Awareness Gate if retrieval is necessary
        if hasattr(self, "cag") and not self.cag.should_retrieve(query_str):
            return {
                **self.classify_context_rule_based(applicant_profile),
                "timestamp": datetime.now().isoformat()
            }

        # Retrieval is required and classify using LLM
        result = self.classify_context_with_llm(applicant_profile)
        result["timestamp"] = datetime.now().isoformat()
        return result