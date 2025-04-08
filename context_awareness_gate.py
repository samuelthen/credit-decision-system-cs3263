from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContextAwarenessGate:
    """
    Context Awareness Gate (CAG) for Retrieval-Augmented Generation (RAG) systems.

    Purpose:
    This gate determines whether a given user query requires external retrieval
    (e.g., from a document store or knowledge base) before answering.

    CAG uses a hybrid approach:
    - Statistical decision using cosine similarity between the query and pseudo-document embeddings.
    - Optional override via an LLM-based yes/no classification signal.

    This mechanism improves computational efficiency and reduces hallucination risks
    by skipping unnecessary retrievals for simple/general queries.
    """
    def __init__(self, embedder, documents, threshold=0.35, top_k=3, use_llm_signal=False, llm_client=None,
                 debug=False):
        """
        Initialize the Context Awareness Gate.

        Args:
        embedder: Sentence transformer model used for embedding queries and documents.
        documents: A dictionary of labeled context documents (e.g., { 'bleak': [...], 'neutral': [...], ... }).
        threshold: Cosine similarity threshold above which retrieval is triggered.
        top_k: Number of top similarities to average for decision-making.
        use_llm_signal: Whether to use an LLM to provide a semantic override signal.
        llm_client: LLM API client (e.g., OpenAI client) to invoke for decision-making if enabled.
        debug: If True, prints internal decision logic for transparency and tuning.
        """
        self.embedder = embedder
        self.threshold = threshold
        self.top_k = top_k
        self.use_llm_signal = use_llm_signal
        self.llm_client = llm_client
        self.debug = debug

        # Step 1: Flatten all provided context documents into a list
        self.doc_texts = []
        for doc_list in documents.values():
            self.doc_texts.extend(doc_list)

        # Step 2: Embed all pseudo-documents for similarity comparison
        self.doc_embeddings = self.embedder.encode(self.doc_texts, convert_to_numpy=True)

    def should_retrieve(self, query):
        """
        Main method to determine whether to perform external retrieval for a given query.
        Uses cosine similarity between the query and pseudo-docs, along with optional LLM override.

        Args:
            query (str): The user's input query.
        Returns:
            bool: True if retrieval should be triggered, False if the LLM or local knowledge is sufficient.
        """
        # Embed the user query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)

        # Compute cosine similarities with all document embeddings
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]

        # Take average of top-K most similar documents
        top_k_similarities = sorted(similarities, reverse=True)[:self.top_k]
        avg_top_k = np.mean(top_k_similarities)

        # Debug logging for analysis and tuning
        if self.debug:
            print(f"[CAG] Query: {query}")
            print(f"[CAG] Top-{self.top_k} avg similarity: {avg_top_k:.3f} (Threshold: {self.threshold})")

        # Initial decision based on statistical similarity
        decision = avg_top_k > self.threshold

        # LLM answers the query if needed
        if self.use_llm_signal and self.llm_client:
            llm_decision = self.query_llm_for_signal(query)
            decision = decision or llm_decision # Combine both logic paths
            if self.debug:
                print(f"[CAG] LLM decision: {llm_decision}")

        return decision

    def query_llm_for_signal(self, query):
        """
        Ask the LLM whether this query requires document retrieval.
        This semantic override complements the cosine similarity logic and helps in edge cases.

        Args:
            query (str): The user's input query.

        Returns:
            bool: True if LLM determines retrieval is needed, otherwise False.
        """
        try:
            prompt = f"""
            Does the following query require retrieving external economic context to answer it properly?
            Query: "{query}"
            Return "yes" or "no" only.
            """

            # Call LLM to get a "yes" or "no" answer
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
            return False # Assume retrieval is not needed