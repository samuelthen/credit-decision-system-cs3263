from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContextAwarenessGate:
    """
    Context Awareness Gate (CAG) for Retrieval-Augmented Generation (RAG) systems.

    This module implements the Vector Candidates (VC) algorithm as described by Heydari et al. (2025),
    to decide whether an external retrieval step is necessary for a given user query.

    It combines:
    - Statistical scoring based on cosine similarity between the query and pre-encoded pseudo-documents.
    - An optional override mechanism using a language model (LLM) for ambiguous or edge cases.

    The CAG improves answer quality, efficiency, and reduces hallucinations by avoiding unnecessary retrievals.
    """

    def __init__(self, embedder, documents, threshold=0.35, top_k=3, use_llm_signal=False, llm_client=None, debug=False):
        """
        Initialize the Context Awareness Gate with document embeddings and decision logic.

        Args:
            embedder: Pretrained sentence transformer used to encode queries and documents.
            documents: A dictionary of context-labeled documents, e.g. { 'bleak': [...], 'neutral': [...], ... }.
            threshold: Decision threshold for cosine similarity (used as a confidence score).
            top_k: Number of top matching documents used in the average similarity score.
            use_llm_signal: If True, queries are additionally evaluated using a semantic signal from an LLM.
            llm_client: Optional LLM API client used for yes/no retrieval override.
            debug: If True, prints internal similarity scores and decisions for transparency and tuning.
        """
        self.embedder = embedder
        self.threshold = threshold
        self.top_k = top_k
        self.use_llm_signal = use_llm_signal
        self.llm_client = llm_client
        self.debug = debug

        # Flatten all documents across context categories
        self.doc_texts = []
        for doc_list in documents.values():
            self.doc_texts.extend(doc_list)

        # Precompute embeddings for pseudo-documents
        self.doc_embeddings = self.embedder.encode(self.doc_texts, convert_to_numpy=True)

    def should_retrieve(self, query):
        """
        Determine whether a query should trigger external document retrieval.

        The decision is based on:
        - Cosine similarity between the query and precomputed document embeddings
        - Average similarity of the Top-K most relevant pseudo-documents
        - Optional override from a semantic LLM classification

        Args:
            query (str): The input user query.

        Returns:
            bool: True if retrieval should be triggered, False if the system can answer without it.
        """
        # Embed the incoming query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)

        # Compute similarity between query and each pseudo-document
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]

        # Aggregate Top-K most relevant similarities
        top_k_similarities = sorted(similarities, reverse=True)[:self.top_k]
        avg_top_k = np.mean(top_k_similarities)

        # Normalize similarity to get a confidence score [0.0, 1.0]
        confidence_score = float(np.clip(avg_top_k, 0.0, 1.0))

        # Print detailed debug output if enabled
        if self.debug:
            print(f"[CAG] Query: {query}")
            print(f"[CAG] Top-{self.top_k} avg similarity: {avg_top_k:.3f} (Threshold: {self.threshold})")
            print(f"[CAG] Confidence score: {confidence_score:.3f}")

        # Base decision: retrieve only if confidence exceeds the defined threshold
        decision = confidence_score > self.threshold

        # Optionally ask the LLM to override the statistical decision
        if self.use_llm_signal and self.llm_client:
            llm_decision = self.query_llm_for_signal(query)
            if llm_decision and not decision:
                # Boost the confidence to slightly above threshold to force retrieval
                confidence = self.threshold + 0.05
                decision = True
                if self.debug:
                    print(f"[CAG] LLM override activated. Boosted confidence to {confidence:.3f}")
            elif self.debug:
                print(f"[CAG] LLM decision: {llm_decision}")

        return decision

    def query_llm_for_signal(self, query):
        """
        Ask a language model if external retrieval is needed to answer the query.

        The LLM is prompted with a yes/no question and can override the statistical decision,
        especially for semantic nuance that is not captured by embedding similarity.

        Args:
            query (str): The user's natural language query.

        Returns:
            bool: True if the LLM recommends performing retrieval, otherwise False.
        """
        try:
            prompt = f"""
            Does the following query require retrieving external economic context to answer it properly?
            Query: "{query}"
            Return "yes" or "no" only.
            """

            # Perform an API call to the LLM (gpt-4o-mini) to get a yes/no answer
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0,
                max_tokens=5
            )
            answer = response.choices[0].message.content.strip().lower()
            return "yes" in answer

        except Exception as e:
            print(f"⚠️ LLM signal error: {e}")
            return False  # assume retrieval not required