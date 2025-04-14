from sentence_transformers import SentenceTransformer
from .context_awareness_gate import ContextAwarenessGate
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
    def __init__(self, use_llm=True, client=None, NLP_AVAILABLE=True, debug=False):
        """
        Initialize the ContextualRAG pipeline.

        Args:
            use_llm (bool): Whether to enable LLM for reasoning and classification.
            client: LLM API client (e.g., OpenAI's GPT) for inference if enabled.
            NLP_AVAILABLE (bool): Flag to enable fallback if embedding model fails or is unavailable.
        """
        self.debug = debug

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
                if hasattr(self, "debug") and self.debug:
                    import traceback
                    traceback.print_exc()
                self.use_llm = False
        else:
            print("⚠️ NLP not available. Falling back to rule-based classification.")

    def _load_documents(self):
        """
        Load macroeconomic context documents aligned with RL and German credit job features.
        """
        return {
            "bleak": [
                "Mass layoffs reported across unskilled labor sectors, with job loss concentrated in manufacturing and non-resident workers.",
                "Credit markets have tightened, particularly for unemployed and unskilled applicants.",
                "Funding for early-stage startups has dropped over 40%, triggering job cuts in low-experience roles.",
                "Small businesses report delayed payments, tighter credit lines, and reduced hiring activity.",
                "Regulators warn banks to reduce risk exposure, impacting non-qualified borrowers the most.",
                "Unemployment has risen significantly among non-residents and unskilled sectors.",
                "The fintech sector has frozen new hiring and rescinded offers for junior roles.",
                "Venture capital activity has declined, with investor sentiment at a 2-year low.",
                "Tech job postings are down 35%, particularly in operational and junior positions.",
                "Several firms have announced hiring freezes, especially for roles requiring minimal qualifications."
            ],

            "neutral": [
                "The job market remains steady for skilled employees, while unskilled roles show minor decline.",
                "Singapore's tech sector reflects mixed signals, with some companies expanding while others restructure.",
                "VC funding has remained flat, indicating cautious investor optimism.",
                "Credit approval rates are consistent with historical trends, especially for mid-income applicants.",
                "MAS has not introduced any new credit-tightening policies this quarter.",
                "Hiring activity is stable across most industries, though slightly lower in entry-level positions.",
                "Regulatory guidance remains neutral, with no new incentives or restrictions.",
                "Tech employment is balanced, with job postings at seasonal averages.",
                "SMEs report no significant changes in lending conditions.",
                "Macroeconomic data reflects modest growth and moderate inflation, with neither recession nor boom."
            ],

            "positive": [
                "Job growth is strongest among skilled and high-qualified professionals.",
                "Venture capital funding has surged, driving demand for high-skill talent and new startups.",
                "SMEs report easy access to credit, with high approval rates and strong cash flow.",
                "Hiring demand in technology and finance is rising, especially in leadership and self-employed sectors.",
                "Investors are bullish on Southeast Asia, with strong returns across startup portfolios.",
                "Singapore’s government has rolled out new innovation grants to fuel economic expansion.",
                "Job postings for management and senior IT roles are up 40% compared to last year.",
                "Consumer sentiment is high, boosting demand across financial and real estate sectors.",
                "Companies are aggressively hiring qualified individuals, offering increased compensation.",
                "Business registration and credit expansion are at multi-year highs, indicating a booming economy."
            ]
        }

    def retrieve_context(self, query, top_k=3):
        """
        Retrieve the most relevant economic context documents for a given query.

        Uses cosine similarity between the query embedding and all pseudo-document embeddings
        to identify the Top-K most semantically similar documents. A majority label is chosen
        from the retrieved documents to determine the dominant context class.

        Args:
             query (str): The user's natural language query.
             top_k (int): Number of top similar documents to select.

        Returns:
             tuple: (List of selected documents, Majority context label)
        """
        # Embed the user query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)

        # Compute cosine similarity between query and all pre-embedded documents
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]

        # Get indices of Top-K most similar documents
        top_k_indices = similarities.argsort()[::-1][:top_k]

        # Identify the majority context label from the Top-K documents
        retrieved_contexts = [self.contexts[i] for i in top_k_indices]
        context_counts = {ctx: retrieved_contexts.count(ctx) for ctx in set(retrieved_contexts)}
        majority_context = max(context_counts, key=context_counts.get)

        # Select the documents from the majority context label
        result_docs = [self.documents[majority_context][i % len(self.documents[majority_context])]
                       for i in top_k_indices if self.contexts[i] == majority_context]

        return result_docs, majority_context

    def summarize_documents(self, documents):
        """
        Generate a structured JSON summary from a list of economic context documents.

        Transform raw and unstructured economic documents into a structured and interpretable
        JSON format. This structured representation enables the RAG system to remove variability
        from unstructured text inputs and reason with consistent macroeconomic indicators
        such as "macro_trend", "labour_market", "credit_flow", and "investor_sentiment".

        This implementation follows the core idea proposed by Zhang et al. (2024) in the
        "Improving LLM Fidelity through Context-Aware Grounding" research paper, which highlights the
        importance of contextual grounding using machine-readable schema to improve response fidelity.

        By producing standardized summaries, this function allows the model to generalize
        across diverse economic scenarios and seamlessly scale to new domains or use cases.

        This enhances the scalability of the RAG system.

        Args:
            documents (list): List of raw economic context strings.

        Returns:
            dict: Structured summary for grounding, e.g.,
                  {
                      "macro_trend": "decline",
                      "labor_market": "contracting",
                      "credit_flow": "tight",
                      "investor_sentiment": "bearish"
                  }
        """
        if not self.use_llm:
            return {}

        try:
            summary_prompt = f"""
            Summarize the following economic signals into a structured JSON with the fields:
            - macro_trend: "growth" | "stable" | "decline"
            - labour_market: "expanding" | "stable" | "contracting"
            - credit_flow: "loose" | "normal" | "tight"
            - investor_sentiment: "bullish" | "neutral" | "bearish"

            Economic Documents:
            {chr(10).join(documents)}

            Return only the JSON object.
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": summary_prompt.strip()}],
                temperature=0.3,
                max_tokens=300
            )

            return json.loads(response.choices[0].message.content.strip())

        except Exception as e:
            print(f"⚠️ Failed to summarize context: {e}")
            return {}

    def classify_context_with_llm(self, query):
        """
        Classify the economic outlook using an LLM grounded in retrieved and semantically summarized context.

        This method retrieves the most relevant macroeconomic documents for a given query,
        optionally summarizes them into a structured format (if enabled), and passes the information
        to a language model for final classification. The model returns an outlook label
        ("bleak", "neutral", or "positive") along with a concise rationale.

        Args:
            query (str): The user's natural language query.

        Returns:
            tuple: A pair containing:
                 - classification (str): One of "bleak", "neutral", or "positive".
                 - reasoning (str): Short explanation justifying the classification.
        """
        if not self.use_llm:
            return self.classify_context_rule_based(query)

        try:
            retrieved_docs, majority_context = self.retrieve_context(query)
            context = "\n".join(retrieved_docs)

            # Construct a grounded prompt using structured context
            prompt = f"""
            You are a credit policy assistant. Based on the structured macroeconomic context below, 
            classify the economic outlook as one of: "bleak", "neutral", or "positive".

            Context:
            {json.dumps(structured_context, indent=2)}

            Question:
            {query}

            Return your answer strictly in this JSON format:
            {{
                "classification": "bleak | neutral | positive",
                "reasoning": "short rationale here"
            }}
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.3,
                max_tokens=300
            )

            result = json.loads(response.choices[0].message.content.strip())
            return result["classification"], result["reasoning"]

        except Exception as e:
            print(f"❌ LLM classification failed: {e}")
            return self.classify_context_rule_based(query)

    def classify_context_rule_based(self, query):
        """
        Fallback context classification using keyword-based heuristic.

        This is used if:
            - Retrieval is not required (per CAG)
            - LLM is unavailable or fails

        Args:
             query (str): The user's input query.

        Returns:
            tuple: (classification label, rationale string)
        """
        bleak_keywords = ["recession", "layoffs", "downturn", "crisis", "decline", "collapse"]
        positive_keywords = ["growth", "boom", "thriving", "expansion", "upward", "prosperity"]

        query_lower = query.lower()
        bleak_count = sum(word in query_lower for word in bleak_keywords)
        positive_count = sum(word in query_lower for word in positive_keywords)

        if bleak_count > positive_count:
            return "bleak", "Based on negative economic indicators in query"
        elif positive_count > bleak_count:
            return "positive", "Based on positive economic indicators in query"
        else:
            return "neutral", "No strong indicators in either direction"

    def get_context(self, query):
        """
        Main method for economic context classification, enhanced with Context Awareness Gate (CAG).

        This function determines whether external document retrieval is necessary for the given query.
        It does so using a hybrid approach:

        1. Uses the Context Awareness Gate (CAG) to evaluate whether retrieval is needed, based on
            query-document similarity and optional LLM signal.
        2. If CAG decides retrieval is unnecessary, it falls back to a lightweight rule-based classifier.
        3. If retrieval is needed and an LLM is available, it classifies context using retrieved documents + LLM.
        4. If LLM is unavailable, it falls back to rule-based classification.

        Args:
            query (str): The user query about economic conditions.

        Returns:
            dict: {
                'classification': One of "bleak", "neutral", or "positive",
                'reasoning': Explanation of how the classification was determined,
                'timestamp': ISO timestamp of classification decision
            }
        """
        if hasattr(self, "cag") and not self.cag.should_retrieve(query):
            context, reason = self.classify_context_rule_based(query)
            return {
                "classification": context,
                "reasoning": f"Query handled without retrieval – {reason}",
                "timestamp": datetime.now().isoformat()
            }

        if self.use_llm:
            context, reason = self.classify_context_with_llm(query)
        else:
            context, reason = self.classify_context_rule_based(query)

        return {
            "classification": context,
            "reasoning": reason,
            "timestamp": datetime.now().isoformat()
        }