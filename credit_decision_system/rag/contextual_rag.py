from sentence_transformers import SentenceTransformer
import json
import faiss
from datetime import datetime

class ContextualRAG:
    """
    Retrieval-Augmented Generation system to determine economic context
    """
    def __init__(self, use_llm=True, client=None, NLP_AVAILABLE=True):
        self.use_llm = use_llm and client is not None
        self.client = client
        
        # Define economic context documents (bleak, neutral, positive)
        self.documents = {
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
        
        # Set up embedding model if available
        if NLP_AVAILABLE:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
                
                # Flatten all documents with context labels
                all_docs = []
                all_contexts = []
                for context, docs in self.documents.items():
                    all_docs.extend(docs)
                    all_contexts.extend([context] * len(docs))
                
                # Create embeddings
                self.doc_embeddings = self.embedder.encode(all_docs, convert_to_numpy=True)
                self.contexts = all_contexts
                
                # Set up FAISS index
                dimension = self.doc_embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(self.doc_embeddings)
                
                print("✅ RAG system initialized successfully")
            except Exception as e:
                print(f"❌ Error setting up RAG: {e}")
                self.use_llm = False
        else:
            print("⚠️ Using rule-based context classification (RAG unavailable)")
    
    def retrieve_context(self, query, top_k=3):
        """Retrieve most relevant documents for a query"""
        
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        
        retrieved_docs = [self.doc_embeddings[i] for i in indices[0]]
        retrieved_contexts = [self.contexts[i] for i in indices[0]]
        
        # Count contexts to find most common
        context_counts = {}
        for context in retrieved_contexts:
            context_counts[context] = context_counts.get(context, 0) + 1
        
        # Get majority context
        majority_context = max(context_counts.items(), key=lambda x: x[1])[0]
        
        # Get the documents corresponding to the majority context
        result_docs = [self.documents[majority_context][i % len(self.documents[majority_context])] 
                      for i, ctx in enumerate(retrieved_contexts) if ctx == majority_context]
        
        return result_docs, majority_context
    
    def classify_context_with_llm(self, query):
        """Use LLM to classify context based on retrieved documents"""
        if not self.use_llm:
            return self.classify_context_rule_based(query)
        
        try:
            retrieved_docs, majority_context = self.retrieve_context(query)
            context = "\n".join(retrieved_docs)
            
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
            
            try:
                result = json.loads(raw_text)
                return result["classification"], result["reasoning"]
            except json.JSONDecodeError:
                print("⚠️ Failed to parse LLM response as JSON")
                return majority_context, "Based on document similarity"
                
        except Exception as e:
            print(f"❌ Error in LLM classification: {e}")
            return self.classify_context_rule_based(query)
    
    def classify_context_rule_based(self, query):
        """Rule-based fallback when LLM is unavailable"""
        # Simple keyword-based classification
        bleak_keywords = ["recession", "layoffs", "downturn", "crisis", "decline", "collapse"]
        positive_keywords = ["growth", "boom", "thriving", "expansion", "upward", "prosperity"]
        
        query_lower = query.lower()
        
        bleak_count = sum(1 for word in bleak_keywords if word in query_lower)
        positive_count = sum(1 for word in positive_keywords if word in query_lower)
        
        if bleak_count > positive_count:
            return "bleak", "Based on negative economic indicators in query"
        elif positive_count > bleak_count:
            return "positive", "Based on positive economic indicators in query"
        else:
            return "neutral", "No strong indicators in either direction"
    
    def get_context(self, query):
        """Main method to get economic context classification"""
        if self.use_llm:
            context, reasoning = self.classify_context_with_llm(query)
        else:
            context, reasoning = self.classify_context_rule_based(query)
            
        return {
            "classification": context,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }