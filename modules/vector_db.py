from sentence_transformers import SentenceTransformer
import os
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

class VectorDB:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY is missing in environment")

        # Pinecone v8 client
        self.pc = Pinecone(api_key=api_key)

        self.index_name = "enterprise-rag-index"

        # Get existing indexes (v8 returns objects)
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        # Create index if missing
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-west-2"
                    }
                }
            )

        # Connect to index
        self.index = self.pc.Index(self.index_name)

        # Embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def upsert_documents(self, documents):
        vectors = []

        for doc in documents:
            text = doc["text"]
            doc_id = doc.get("id") or str(uuid.uuid4())
            embedding = self.model.encode(text).tolist()

            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": {"text": text}
            })

        self.index.upsert(vectors=vectors)

    def query(self, query_text, top_k=5):
        embedding = self.model.encode(query_text).tolist()

        results = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )

        matches = []
        for m in results.get("matches", []):
            matches.append({
                "id": m["id"],
                "score": m["score"],
                "metadata": m.get("metadata", {})
            })

        return {"matches": matches}
