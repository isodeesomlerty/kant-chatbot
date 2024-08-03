import numpy as np
import faiss
from typing import List, Dict, Any
from config import config

class KantRetriever:
    def __init__(self, processed_data: List[Dict[str, Any]], embeddings: np.ndarray):
        self.processed_data = processed_data
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query_embedding: np.ndarray, k: int = config['num_relevant_chunks']) -> List[Dict[str, Any]]:
        if not isinstance(query_embedding, np.ndarray):
            raise TypeError("query_embedding must be a numpy array")
        
        if query_embedding.ndim != 1:
            raise ValueError("query_embedding must be a 1-dimensional array")
        
        query_embedding = query_embedding.reshape(1, -1)  # Ensure it's 2D for faiss
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i in indices[0]:
            results.append(self.processed_data[i])
        
        return results

def create_retriever(processed_data: List[Dict[str, Any]], embeddings: np.ndarray) -> KantRetriever:
    return KantRetriever(processed_data, embeddings)

if __name__ == "__main__":
    from embedding import create_embeddings, get_embeddings
    from data_preprocessing import process_urls
    from config import get_kant_urls
    
    kant_urls = get_kant_urls()
    processed_data = process_urls(kant_urls)
    embeddings = create_embeddings(processed_data)
    
    retriever = create_retriever(processed_data, embeddings)
    
    # Test query
    test_query = "What is Kant's view on ethics?"
    query_embedding = np.array(get_embeddings([test_query])[0])
    
    results = retriever.retrieve(query_embedding)
    
    print(f"Top {len(results)} results for query: '{test_query}'")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['text'][:100]}... (Source: {result['source']})")