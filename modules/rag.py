# modules/rag.py
from modules.vector_db import VectorDB
from modules.llm_interface import LLMInterface

class RAG:
    def __init__(self, llm):
        self.vector_db = VectorDB()
        self.llm = llm

    def answer_dataset_question(self, question, dataset):
        """Answer questions related to the dataset."""
        context = (
            "You are an expert data analyst. Use the dataset summary below to provide a detailed answer to the user's question.\n\n"
            f"Dataset Summary:\n{dataset.describe(include='all').to_string()}\n\n"
            "Value Counts for Categorical Columns (Top 10):\n"
        )
        categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            counts = dataset[col].value_counts().head(10).to_string()
            context += f"\nColumn '{col}':\n{counts}\n"
        context += f"\nDataset Columns: {', '.join(dataset.columns)}\n"
        context += "Remember to reference specific columns and data points in your answer.\n\n"
        context += f"Question: {question}"
        response = self.llm.conversational_response([{'sender': 'user', 'text': context}])
        return response

    def answer_question(self, question):
        search_results = self.vector_db.query(question)
        context = ""
        for match in search_results.get('matches', []):
            context += match['metadata']['text'] + "\n"
        if not context:
            context = "No relevant context found."
        conversation = [
            {"sender": "user", "text": f"Based on the following context, provide a detailed answer to the question.\n\nContext:\n{context}\n\nQuestion: {question}"}
        ]
        response = self.llm.conversational_response(conversation)
        return response

    def log_interaction(self, user_query, ai_response):
        try:
            with open("interaction_log.txt", "a") as log_file:
                log_file.write(f"User Query: {user_query}\nAI Response: {ai_response}\n\n")
        except Exception as e:
            print(f"Error logging interaction: {str(e)}")
