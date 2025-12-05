# AI-Powered-Strategic-Navigator-for-Business

AI-powered decision-making platform that integrates Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to provide real-time insights for businesses. This platform enables enterprises to make faster, data-driven decisions by combining internal data with external market trends.

## 🆕 New Features

### 🔧 IFB Service Ecosystem Forecasting & Analytics
Comprehensive AI-driven forecasting and analytics system for IFB's nationwide service network:
- **30/60/90-day demand forecasting** for service volumes, spare parts, and warranty claims
- **Inventory optimization** with ABC analysis and location-specific insights
- **Franchise performance tracking** with transparent scorecards and reports
- **Revenue optimization** identifying and reducing service-specific leakages
- **ML-powered predictions** using Gradient Boosting and Random Forest models
- **Location intelligence** across branches, regions, and franchises

### 💸 Revenue Leakage Analysis
Advanced revenue leakage detection for any business:
- **8 comprehensive analysis modules** covering discounts, profit erosion, products, regions
- **AI-powered forecasting** to predict future revenue leakages
- **Anomaly detection** using Isolation Forest algorithms
- **Actionable recommendations** for immediate, short-term, and long-term improvements


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

## 🔧 IFB Service Forecasting Setup

Generate sample IFB service ecosystem data:
```bash
python generate_ifb_sample_data.py
```

This creates realistic service data including:
- 5,000+ service records
- 20 locations across India
- 11 franchise partners
- Complete service lifecycle data

See `IFB_SERVICE_FORECASTING_GUIDE.md` for detailed documentation.
