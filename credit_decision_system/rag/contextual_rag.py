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

    def retrieve_context(self, query, top_k=3):
        """
        Retrieve the most relevant economic context documents for a given query.

        Uses vector similarity search over pre-embedded documents and selects documents
        based on majority label voting from Top-K nearest neighbors.

        Args:
             query (str): The user's natural language query.
             top_k (int): Number of nearest neighbors to retrieve from the FAISS index.

        Returns:
             tuple: (List of selected documents, Majority context label)
        """
        # Optionally enrich the query with related terms (not used in embedding here)
        expanded_query = f"{query} Related terms: market, finance, investment, tech, economy"
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)

        # Determine which context label is most represented among the top documents
        retrieved_contexts = [self.contexts[i] for i in indices[0]]
        context_counts = {ctx: retrieved_contexts.count(ctx) for ctx in set(retrieved_contexts)}
        majority_context = max(context_counts, key=context_counts.get)

        # Return documents from the winning context label
        result_docs = [self.documents[majority_context][i % len(self.documents[majority_context])]
                       for i, ctx in enumerate(retrieved_contexts) if ctx == majority_context]

        return result_docs, majority_context

    def classify_context_with_llm(self, query):
        """
        Classify the economic outlook using an LLM, based on documents retrieved from FAISS.

        This is triggered only if CAG determines that retrieval is necessary.

        Args:
             query (str): The user's natural language query.

        Returns:
              tuple: (classification label, rationale string)
        """
        if not self.use_llm:
            return self.classify_context_rule_based(query)

        try:
            retrieved_docs, majority_context = self.retrieve_context(query)
            context = "\n".join(retrieved_docs)

            # Construct prompt for LLM-based classification
            prompt = f"""
            You are a credit policy assistant. Based on the following macroeconomic context, 
            classify the economic outlook as one of: "bleak", "neutral", or "positive".

            Return your answer strictly in this JSON format:
            {{
              "classification": "bleak | neutral | positive",
              "reasoning": "short rationale here"
            }}

            Context:
            {context}

            Question:
            {query}
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.3,
                max_tokens=300
            )

            raw_text = response.choices[0].message.content.strip()
            result = json.loads(raw_text)
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
        # Check with Context Awareness Gate if retrieval is necessary
        if hasattr(self, "cag") and not self.cag.should_retrieve(query):
            context, reason = self.classify_context_rule_based(query)
            return {
                "classification": context,
                "reasoning": f"Query handled without retrieval – {reason}",
                "timestamp": datetime.now().isoformat()
            }

        # Retrieval is required and classify using LLM if available
        if self.use_llm:
            context, reason = self.classify_context_with_llm(query)
        else:
            # Fallback: rule-based if LLM is not available
            context, reason = self.classify_context_rule_based(query)

        # Return classification with reasoning and timestamp
        return {
            "classification": context,
            "reasoning": reason,
            "timestamp": datetime.now().isoformat()
        }