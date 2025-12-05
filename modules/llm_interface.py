import os
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

class LLMInterface:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing in environment")

        self.client = OpenAI(api_key=api_key)
        self.model_name = "gpt-4o-mini"  # or 4.1, gpt-4o, etc.

    def conversational_response(self, conversation):
        """Generates a response with OpenAI new SDK format."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful data analyst assistant who provides detailed and specific "
                        "answers based on the dataset. Never provide code. If asked for charts, "
                        "describe insights instead."
                    )
                }
            ]

            for msg in conversation:
                messages.append({"role": "user", "content": msg["text"]})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=7048,
                temperature=0.2,
            )

            ai_response = response.choices[0].message.content
            return {"text": ai_response, "confidence": 0.90}

        except OpenAIError as e:
            return {"text": f"Error: {str(e)}", "confidence": 0.0}

    def generate_strategic_recommendations(self, data_summary):
        prompt = (
            "As a seasoned business strategist, analyze the following data and offer detailed, "
            "actionable strategies.\n\n"
            f"Data Summary:\n{data_summary}\n\n"
            "Provide a comprehensive analysis and strategic recommendations."
        )
        return self.conversational_response([{"sender": "user", "text": prompt}])["text"]

    def generate_risk_analysis(self, strategies):
        prompt = (
            "For each of the following strategies, perform a risk analysis. Identify potential risks "
            "and suggest mitigation plans.\n\n"
            f"Strategies:\n{strategies}\n\n"
            "Provide the risk analysis in bullet points under each strategy."
        )
        return self.conversational_response([{"sender": "user", "text": prompt}])["text"]

    def estimate_resources(self, strategies):
        prompt = (
            "Estimate the resources required to implement each of the following strategies. Include time, "
            "budget, and personnel estimates.\n\n"
            f"Strategies:\n{strategies}\n\n"
            "Provide the estimates clearly."
        )
        return self.conversational_response([{"sender": "user", "text": prompt}])["text"]

    def simulate_custom_strategy(self, strategy_input, data_summary):
        prompt = (
            f"Given the data summary:\n{data_summary}\n\n"
            f"Simulate the potential outcomes of implementing this strategy:\n{strategy_input}\n\n"
            "Provide a detailed impact analysis on key business metrics."
        )
        return self.conversational_response([{"sender": "user", "text": prompt}])["text"]

    def generate_response(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a highly knowledgeable assistant. Do not provide code."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=7098,
                temperature=0.2
            )
            return response.choices[0].message["content"]

        except OpenAIError as e:
            return f"Error generating response: {str(e)}"
