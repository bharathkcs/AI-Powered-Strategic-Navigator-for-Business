import openai
import os
from dotenv import load_dotenv

load_dotenv()

class LLMInterface:
    def __init__(self):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.model_name = "gpt-4o-mini"  # Use gpt-3.5-turbo or gpt-4o-mini

    def conversational_response(self, conversation):
        """Generates a response based on the conversation."""
        try:
            messages = [{"role": "system", "content": "You are a helpful data analyst assistant who provides detailed and specific answers based on the provided dataset. Do not provide code in your responses. If the user asks for a plot or chart, describe the insights instead of providing code."}]
            for msg in conversation:
                messages.append({"role": "user", "content": msg["text"]})

            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                max_tokens=7048,  
                temperature=0.2  
            )
            ai_response = response['choices'][0]['message']['content']
            return {"text": ai_response, "confidence": 0.90}
        except openai.error.OpenAIError as e:
            return {"text": f"Error: {str(e)}", "confidence": 0.0}

    def generate_strategic_recommendations(self, data_summary):
        """Generates strategic recommendations based on data summary."""
        prompt = (
            f"As a seasoned business strategist, analyze the following data and offer detailed, actionable strategies.\n\n"
            f"Data Summary:\n{data_summary}\n\n"
            "Provide a comprehensive analysis and strategic recommendations."
        )

        response = self.conversational_response([{'sender': 'user', 'text': prompt}])
        return response['text']

    def generate_risk_analysis(self, strategies):
        """Generates a risk analysis for each strategy."""
        prompt = (
            f"For each of the following strategies, perform a risk analysis. Identify potential risks and suggest mitigation plans.\n\n"
            f"Strategies:\n{strategies}\n\n"
            "Provide the risk analysis in bullet points under each strategy."
        )

        response = self.conversational_response([{'sender': 'user', 'text': prompt}])
        return response['text']

    def estimate_resources(self, strategies):
        """Estimates resources required for each strategy."""
        prompt = (
            f"Estimate the resources required to implement each of the following strategies. Include time, budget, and personnel estimates.\n\n"
            f"Strategies:\n{strategies}\n\n"
            "Provide the estimates in a clear and concise manner."
        )

        response = self.conversational_response([{'sender': 'user', 'text': prompt}])
        return response['text']

    def simulate_custom_strategy(self, strategy_input, data_summary):
        """Simulates the potential outcomes of a custom strategy."""
        prompt = (
            f"Given the following data summary:\n{data_summary}\n\n"
            f"Simulate the potential outcomes of implementing the following strategy:\n{strategy_input}\n\n"
            "Provide a detailed analysis of the expected impact on key business metrics."
        )

        response = self.conversational_response([{'sender': 'user', 'text': prompt}])
        return response['text']
    
    def process_question(self, question):
        """Processes the user's question and returns an answer."""
        
        data_summary = self.data.describe(include='all').transpose().to_string()
        
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        value_counts = ""
        for col in categorical_cols:
            counts = self.data[col].value_counts().head(10).to_string()
            value_counts += f"\nColumn '{col}' value counts:\n{counts}\n"
        
        max_length = 1500 
        total_length = len(data_summary) + len(value_counts)
        if total_length > max_length:
            data_summary = data_summary[:max_length//2] + "\n... [Data truncated]"
            value_counts = value_counts[:max_length//2] + "\n... [Data truncated]"
        prompt = (
            f"You are an expert data analyst. Use the dataset summary below to provide a detailed answer to the user's question.\n\n"
            f"Dataset Summary:\n{data_summary}\n\n"
            f"{value_counts}\n\n"
            "Remember to reference specific columns and data points in your answer.\n\n"
            f"Question: {question}"
        )
        response = self.llm.conversational_response([{'sender': 'user', 'text': prompt}])
        return response['text']
    
    def generate_response(self, prompt):
        """Generates a response based on a prompt."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a highly knowledgeable assistant. Do not provide code in your responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=7098,
                temperature=0.2
            )
            return response['choices'][0]['message']['content']
        except openai.error.OpenAIError as e:
            return f"Error generating response: {str(e)}"
