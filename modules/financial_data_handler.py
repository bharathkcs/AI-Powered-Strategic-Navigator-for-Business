import yfinance as yf
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import requests  
import urllib.parse


nltk.download('vader_lexicon', quiet=True)

class FinancialDataHandler:
    def __init__(self, llm):
        self.sia = SentimentIntensityAnalyzer()
        self.llm = llm  
    
    def get_symbol_from_name(self, company_name):
        try:
            query = urllib.parse.quote(company_name)
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if 'quotes' in data and len(data['quotes']) > 0:
                    symbol = data['quotes'][0]['symbol']
                    return symbol
                else:
                    return None
            else:
                return None
        except Exception as e:
            print(f"Error fetching symbol: {str(e)}")
            return None

    def get_company_financials(self, company_names):
        try:
            if isinstance(company_names, str):
                company_names = [company_names]

            financials_list = []
            for company_name in company_names:
                symbol = self.get_symbol_from_name(company_name)
                if symbol:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    if 'shortName' in info:  
                        financials = {
                            'Symbol': symbol.upper(),
                            'Company Name': info.get('shortName', 'N/A'),
                            'Sector': info.get('sector', 'N/A'),
                            'Industry': info.get('industry', 'N/A'),
                            'Market Cap': info.get('marketCap', 'N/A'),
                            'Enterprise Value': info.get('enterpriseValue', 'N/A'),
                            'Trailing P/E': info.get('trailingPE', 'N/A'),
                            'Forward P/E': info.get('forwardPE', 'N/A'),
                            'PEG Ratio': info.get('pegRatio', 'N/A'),
                            'Price to Sales': info.get('priceToSalesTrailing12Months', 'N/A'),
                            'Price to Book': info.get('priceToBook', 'N/A'),
                            'Profit Margin': info.get('profitMargins', 'N/A'),
                            'Operating Margin': info.get('operatingMargins', 'N/A'),
                            'Return on Assets': info.get('returnOnAssets', 'N/A'),
                            'Return on Equity': info.get('returnOnEquity', 'N/A'),
                            'Revenue': info.get('totalRevenue', 'N/A'),
                            'Gross Profit': info.get('grossProfits', 'N/A'),
                            'EBITDA': info.get('ebitda', 'N/A'),
                            'Net Income': info.get('netIncomeToCommon', 'N/A'),
                        }
                        financials_list.append(financials)
                    else:
                        print(f"No company found with name: {company_name}")
                else:
                    print(f"Could not find symbol for company name: {company_name}")

            if financials_list:
                financials_df = pd.DataFrame(financials_list)
                financials_df.set_index('Symbol', inplace=True)
                return financials_df
            else:
                return None  
        except Exception as e:
            print(f"Error fetching financials: {str(e)}")
            return None

    def get_company_news(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            news_df = pd.DataFrame(news)
            if not news_df.empty:
                news_df = news_df[['title', 'publisher', 'link', 'providerPublishTime']]
                news_df['providerPublishTime'] = pd.to_datetime(news_df['providerPublishTime'], unit='s')
                
                news_df['Sentiment'] = news_df['title'].apply(lambda x: self.sia.polarity_scores(x)['compound'])
                return news_df
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return pd.DataFrame()

    def get_industry_averages(self, industry, metric):
        
        industry_data = {
            'Technology': {
                'Trailing P/E': 25.0,
                'Forward P/E': 20.0,
                'Profit Margin': 0.15,
                'Return on Equity': 0.18,
            },
            'Healthcare': {
                'Trailing P/E': 30.0,
                'Forward P/E': 25.0,
                'Profit Margin': 0.12,
                'Return on Equity': 0.16,
            },
            'Financial Services': {
                'Trailing P/E': 12.0,
                'Forward P/E': 10.0,
                'Profit Margin': 0.20,
                'Return on Equity': 0.15,
            },
            'Consumer Defensive': {
                'Trailing P/E': 18.0,
                'Forward P/E': 16.0,
                'Profit Margin': 0.10,
                'Return on Equity': 0.12,
            },
            
        }
        return industry_data.get(industry, {}).get(metric, 'N/A')

    def get_stock_data(self, symbol, period='1y'):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            return None

    def get_recent_changes(self, symbol):
        
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            if news:
                
                news_texts = [item['title'] + ". " + item.get('summary', '') for item in news]
                combined_text = " ".join(news_texts)
                prompt = f"Extract the key recent changes, strategies, or adaptations that {symbol} has adopted from the following articles:\n\n{combined_text}"
                insights = self.llm.generate_response(prompt)
                return insights
            else:
                return "No recent news articles available to extract insights."
        except Exception as e:
            print(f"Error fetching recent changes: {str(e)}")
            return "Error fetching recent changes."
