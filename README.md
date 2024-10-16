# AI-Powered-Strategic-Navigator-for-Business
 AI-powered decision-making platform that integrates Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to provide real-time insights for businesses. This platform will enable enterprises to make faster, data-driven decisions by combining internal data with external market trends.


1. Clone the repository: 
git clone https://github.com/bharathkcs/AI-Powered-Strategic-Navigator-for-Business  
cd AI-Powered-Strategic-Navigator-for-Business

2. Install dependencies: pip install -r requirements.txt

3. Set up API keys: Create a .env file in the project directory.
Add the following keys: 

OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
HUGGINGFACE_TOKEN=your_huggingface_api_here
ALPHA_VANTAGE_API_KEY=your_alpha_api_here
HUGGINGFACE_TOKEN=your_huggingface_api_here
ALPHA_VANTAGE_API_KEY=your_alpha_api_here

4. Download NLTK data:
Run this script once:
import nltk
nltk.download('vader_lexicon')

5. Run the Streamlit app:
streamlit run app.py
