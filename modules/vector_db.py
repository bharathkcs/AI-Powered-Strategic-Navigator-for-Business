from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

class VectorDB:
    def __init__(self):
    
        self.pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_ENVIRONMENT')
        )
        self.index_name = "enterprise-rag-index"

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384, 
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-west-2')
            )

        self.index = self.pc.Index(self.index_name)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def upsert_documents(self, documents):
        
        vectors = []
        for doc in documents:
            embedding = self.model.encode(doc['text']).tolist()
            vectors.append((doc['id'], embedding, {'text': doc['text']}))
        self.index.upsert(vectors)

    def query(self, query_text, top_k=5):
        embedding = self.model.encode(query_text).tolist()
        try:
            results = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)
            return results
        except Exception as e:
            return {'matches': [], 'error': str(e)}

    def close(self):
    
        self.pc.close() 
