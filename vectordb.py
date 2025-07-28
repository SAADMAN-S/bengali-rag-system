# Import libraries
import json
import pinecone
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import time
from tqdm import tqdm
import hashlib

print("✅ Pinecone imports ready!")

class BengaliVectorStore:
    def __init__(self, pinecone_api_key: str, index_name: str = "bengali-rag-chunks"):
        """
        Initialize the vector store for Bengali text

        Args:
            pinecone_api_key: Your Pinecone API key
            index_name: Name for your Pinecone index
        """
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name

        # Initialize Pinecone with OLD API
        pinecone.init(api_key=pinecone_api_key, environment="us-east-1")

        # Load multilingual embedding model (good for Bengali)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        # Get embedding dimension
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")

        # Initialize index
        self.index = self._setup_index()

    def _setup_index(self):
        """Setup or connect to Pinecone index"""
        try:
            # Check if index exists (OLD API)
            existing_indexes = pinecone.list_indexes()

            if self.index_name in existing_indexes:
                print(f"Connecting to existing index: {self.index_name}")
                index = pinecone.Index(self.index_name)
            else:
                print(f"Creating new index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dim,
                    metric="cosine"
                )
                # Wait for index to be ready
                print("Waiting for index to be ready...")
                while self.index_name not in pinecone.list_indexes():
                    time.sleep(1)

                index = pinecone.Index(self.index_name)

            return index
        except Exception as e:
            print(f"Error setting up index: {e}")
            raise

    # Keep all your other methods exactly the same
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Create embeddings for Bengali/multilingual text"""
        print(f"Creating embeddings for {len(texts)} chunks...")

        all_embeddings = []

        # Process in batches to avoid memory issues
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + batch_size]

            # Create embeddings
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Important for cosine similarity
            )

            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    # ... (keep all your other methods exactly as they are)

    def search_similar_chunks(self, query: str, top_k: int = 5, filter_dict: Dict = None) -> List[Dict]:
        """Search for similar chunks"""
        # Create query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        # Search in Pinecone
        search_results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )

        # Format results
        results = []
        for match in search_results['matches']:
            result = {
                'id': match['id'],
                'score': match['score'],
                'text': match['metadata']['full_text'],
                'page_number': match['metadata'].get('page_number'),
                'section_type': match['metadata'].get('section_type'),
                'word_count': match['metadata'].get('word_count'),
                'metadata': match['metadata']
            }
            results.append(result)

        return results

# Test it
def quick_test():
    PINECONE_API_KEY = "pcsk_2Jqw4c_A9sCqN2EKV3WY7k2KHuL9doLVzjt2PAS3NSkqB6YgToCid4wLpEvB5Vy2u8Yogq"
    INDEX_NAME = "bengali-rag-aparichita"

    try:
        vector_store = BengaliVectorStore(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=INDEX_NAME
        )
        print("✅ Vector store initialized successfully!")
        return vector_store
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Run test
vector_store = quick_test()