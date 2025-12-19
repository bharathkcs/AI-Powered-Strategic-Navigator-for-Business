import streamlit as st
import pandas as pd
import plotly.express as px
from modules.ui_components import (
    display_title,
    display_footer,
    display_navigation
)
from modules.llm_interface import LLMInterface
from modules.rag import RAG
from modules.business_data_handler import DataAnalyzer
from modules.metric_tracker import MetricTracker
from modules.financial_data_handler import FinancialDataHandler
from modules.strategy_map import StrategyMap
from modules.revenue_leakage_detector import RevenueLeakageDetector
from modules.ifb_service_forecasting import IFBServiceForecasting
from modules.spare_parts_planning_page import render_spare_parts_planning_page


llm = LLMInterface()
rag = RAG(llm)
financial_handler = FinancialDataHandler(llm)

st.set_page_config(page_title="IFB Decision Intelligence Hub", layout="wide")


display_title()
selected_page = display_navigation()

def main():
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None

    if 'financial_data' not in st.session_state:
        st.session_state.financial_data = None

    page_to_display = selected_page

    if page_to_display == "🏠 Home":
        st.header("🎉 Welcome to the IFB Decision Intelligence Hub 🎉")
        st.markdown("""
        🚀 **Transform Your Business Decision-Making with AI** 🚀  
        Our AI-powered solution revolutionizes how businesses make decisions by leveraging advanced **Generative AI** and **Retrieval-Augmented Generation (RAG)** technology. This platform is designed to provide comprehensive insights and tailored strategies based on your specific data, empowering you to make informed decisions with precision.

        ### 🌟 **Key Features and Functionalities** 🌟

        1. **🔍 Unlock Insights**  
           **Overview**: Effortlessly analyze complex datasets to uncover hidden trends and patterns in your business data.  
           **Sub-features**:  
           - **📊 Trend Identification**: Automatically highlights seasonal trends, rising or declining sales, and more.
           - **🔄 Pattern Recognition**: Detects recurring patterns in customer behavior, product demand, or regional performance.

        2. **💡 AI-Powered Strategies**  
           **Overview**: Get actionable, data-driven strategies customized to your business needs.  
           **Sub-features**:  
           - **📈 Strategic Recommendations**: Receive tailored suggestions to improve profitability, operational efficiency, or customer engagement.
           - **⚠️ Risk Analysis**: Understand potential risks associated with each strategy and receive mitigation advice.
           - **🛠️ Resource Estimation**: Estimates the required time, budget, and personnel to implement strategies.
           - **Ask**: "What strategies can increase customer retention based on my recent sales data?"

        3. **💬 Interactive Q&A System**  
           **Overview**: Ask questions in plain language and get instant, data-driven answers.  
           **Sub-features**:  
           - **🧠 Smart Q&A**: Upload data, ask questions, and the AI will generate detailed insights specific to your dataset.
           - **🗂️ Conversational History**: Keep track of previous questions and responses to continue where you left off.
           - **Ask**: "What was our top-selling product in the last quarter?"

        4. **📊 Real-Time Visualization**  
           **Overview**: Visualize your business data dynamically with a variety of chart types.  
           **Sub-features**:  
           - **📈 Customizable Graphs**: Generate bar charts, line charts, scatter plots, or pie charts for any business metric.
           - **📊 Interactive Dashboards**: Build real-time dashboards to monitor key business performance metrics.
           - **Ask**: "Can you create a comparison chart for our monthly sales across regions?"

        5. **📈 Impact Tracking**  
           **Overview**: Monitor the effectiveness of AI-driven strategies and track key metrics over time.  
           **Sub-features**:  
           - **📊 KPI Monitoring**: Set up alerts for critical KPIs like sales, customer acquisition cost, or profit margins.
           - **⚠️ Anomaly Detection**: Automatically detect unusual patterns in your metrics, alerting you to take action.
           - **🔄 Strategy Impact**: Analyze the effects of new strategies on your business, from sales to customer retention.
           - **Ask**: "How has our new pricing strategy impacted customer satisfaction?"

        6. **🔄 Auto-Adaptive Business Strategy Maps**  
           **Overview**: Simulate the potential impact of different strategies on your business.  
           **Sub-features**:  
           - **🎯 Strategy Simulation**: Test various business strategies, such as increasing marketing budgets, and visualize their impact on revenue, market share, and customer satisfaction.
           - **🔍 Scenario Planning**: Compare different strategic options to determine the best course of action.
           - **Ask**: "What if we increase our advertising spend by 20% next quarter?"

        7. **📈 Metric Tracking**
           **Overview**: Track and forecast business metrics, detect anomalies, and set performance alerts.
           **Sub-features**:
           - **🔮 Predictive Analytics**: Forecast future performance for key metrics like sales, customer growth, or market trends.
           - **🔍 Correlation Analysis**: Discover correlations between business metrics, such as the link between marketing spend and sales.
           - **🚨 Anomaly Detection**: Identify unusual spikes or drops in your business metrics and receive automatic alerts.

        8. **💸 Revenue Leakage Analysis** *(NEW!)*
           **Overview**: Identify and forecast revenue leakages across your business operations with advanced AI-powered analysis.
           **Sub-features**:
           - **📊 Executive Dashboard**: Get a comprehensive overview of total revenue leakage, potential savings, and key metrics at a glance.
           - **💰 Discount Analysis**: Identify excessive discounting patterns and their impact on profitability across products and categories.
           - **📉 Profit Erosion Detection**: Track profit margin trends over time and identify categories with declining profitability.
           - **📦 Product Performance**: Analyze underperforming products and categories, identifying which ones are causing revenue leakage.
           - **🌍 Regional Analysis**: Discover regional performance issues and identify geographic areas with revenue leakage problems.
           - **🔮 AI Forecasting**: Use machine learning models to forecast future revenue leakage trends and take preventive action.
           - **📈 Anomaly Detection**: Automatically detect anomalous transactions that indicate potential revenue leakage.
           - **🤖 AI Recommendations**: Receive comprehensive, actionable recommendations to reduce revenue leakage and improve profitability.
           - **Ask**: "Where is my business losing money?" or "How can I reduce revenue leakage?"



        ### **🌐 How to Navigate**

        - **🏠 Home**: Get an overview and introduction to the application.
        - **🔍 Q&A System**: Upload data and ask business-related questions.
        - **📊 Data Insights**: Explore automatic analysis and visualizations of your data.
        - **💡 AI-Powered Strategies**: Receive customized strategic recommendations.
        - **📈 Metric Tracking**: Monitor and forecast key business metrics.
        - **🔄 Auto-Adaptive Strategy Maps**: Simulate business strategies and their impact.

        **Experience the future of business decision-making today!** 🚀  
        **Need help?** Reach out to us at 📧 [kcsb28@gmail.com](mailto:kcsb28@gmail.com).

        ### 🔧 **Technical Insights**  
        We’ve leveraged cutting-edge technologies like **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** to provide highly accurate and context-aware insights. The platform is built with **Streamlit** for an interactive, real-time user experience and integrates **Plotly** for dynamic data visualizations. It’s designed to be scalable and adaptable to various business environments, ensuring optimal decision-making.
        """)
        st.image("https://via.placeholder.com/800x300.png/00bfff/ffffff?text=AI-Powered+Strategic+Navigator", use_column_width=True)

    elif page_to_display == "📈 Metric Tracking":
        st.header("Predictive Metric Tracking and Anomaly Detection")
        st.write("Forecast future performance, detect anomalies, and gain insights on key business metrics.")

        if st.session_state.uploaded_data is not None:
            metric_tracker = MetricTracker(llm, st.session_state.uploaded_data)
            metric_tracker.track_metrics()
        else:
            st.write("Please upload a dataset in the Q&A System page to track and forecast metrics.")

    elif page_to_display == "📈 Metric Tracking":
        st.header("Predictive Metric Tracking and Anomaly Detection")
        st.write("Forecast future performance, detect anomalies, and gain insights on key business metrics.")

        if st.session_state.uploaded_data is not None:
            metric_tracker = MetricTracker(llm, st.session_state.uploaded_data)
            metric_tracker.track_metrics()
        else:
            st.write("Please upload a dataset in the Q&A System page to track and forecast metrics.")

    elif page_to_display == "📊 Auto-Adaptive Business Strategy Maps":
        st.header("AI-Powered Scenario Planning Tool")
        st.write("Simulate the potential impact of different strategies on key business metrics.")

        if st.session_state.uploaded_data is not None:
            strategy_map = StrategyMap(st.session_state.uploaded_data, llm)
            strategy_map.display_interactive_scenario()
        else:
            st.error("No dataset uploaded. Please upload a dataset in the Q&A System page to generate strategy maps.")

    elif page_to_display == "💸 Revenue Leakage Analysis":
        st.header("AI-Powered Revenue Leakage Detection & Forecasting")
        st.markdown("""
        This advanced analysis identifies revenue leakages across your business operations,
        forecasts future trends, and provides actionable recommendations to improve profitability.
        """)

        if st.session_state.uploaded_data is not None:
            try:
                revenue_detector = RevenueLeakageDetector(st.session_state.uploaded_data, llm)
                revenue_detector.detect_leakages()
            except Exception as e:
                st.error(f"Error in revenue leakage analysis: {str(e)}")
                st.exception(e)

        else:
            st.warning("⚠️ Please upload a dataset in the Q&A System page to analyze revenue leakages.")
            st.info("""
            **Required Dataset Columns:**
            - Sales (revenue data)
            - Profit (profit data)
            - Discount (discount rates, optional)
            - Order Date (for time-series analysis, optional)
            - Category/Product information (optional)
            - Region/Location information (optional)
            """)

    elif page_to_display == "🔧 IFB Service Forecasting":
        st.header("🔧 IFB Service Ecosystem - AI Forecasting & Analytics")
        st.markdown("""
        **Advanced forecasting and analytics for IFB's nationwide service network**

        This module provides comprehensive 30/60/90-day demand forecasting, inventory optimization,
        franchise performance tracking, and revenue optimization for service operations.
        """)

        # Option to load sample IFB data or use uploaded data
        use_sample_data = st.checkbox("Load IFB Sample Service Data", value=False,
                                      help="Use pre-loaded IFB service ecosystem sample data")

        if use_sample_data:
            try:
                ifb_data = pd.read_csv('data/ifb_service_data.csv')
                st.success(f"✅ Loaded IFB sample data: {len(ifb_data)} service records")
                st.info(f"📅 Date range: {ifb_data['Service_Date'].min()} to {ifb_data['Service_Date'].max()}")

                # Initialize and run IFB forecasting system
                ifb_system = IFBServiceForecasting(ifb_data, llm)
                ifb_system.run_analysis()

            except FileNotFoundError:
                st.error("Sample data file not found. Please run: python generate_ifb_sample_data.py")
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")

        elif st.session_state.uploaded_data is not None:
            st.info("Using your uploaded dataset for IFB service analysis")

            # Check if the data has service-related columns
            required_cols_check = any(col in st.session_state.uploaded_data.columns
                                     for col in ['Service_ID', 'Service_Revenue', 'Service_Type'])

            if required_cols_check:
                try:
                    ifb_system = IFBServiceForecasting(st.session_state.uploaded_data, llm)
                    ifb_system.run_analysis()
                except Exception as e:
                    st.error(f"Error in IFB service analysis: {str(e)}")
            else:
                st.warning("""
                Your dataset doesn't appear to have service-related columns.

                **Expected columns for IFB Service Analysis:**
                - Service_ID, Service_Date, Service_Revenue
                - Location, Branch, Region, Franchise_ID
                - Product_Category, Service_Type
                - Parts_Used, Parts_Cost, Parts_Revenue
                - Warranty_Claim, Technician_ID
                - Customer_Satisfaction, Service_Duration

                Try loading the IFB sample data using the checkbox above.
                """)
        else:
            st.warning("⚠️ No data loaded")
            st.info("""
            **Choose one option:**

            1. ✅ **Check the box above** to load IFB sample service data (recommended for demo)

            2. **Upload your own service data** in the Q&A System page with these columns:
               - Service_ID, Service_Date, Service_Revenue
               - Location, Region, Franchise_ID
               - Product_Category, Service_Type
               - Parts information, Warranty data
               - Customer metrics

            **Sample Data Includes:**
            - 5,000+ service records
            - 20 locations across India
            - 11 franchise partners
            - Multiple product categories
            - Complete service lifecycle data
            """)

    elif page_to_display == "⚙️ Spare Parts Planning":
        # NEW TAB: Spare Parts Demand Forecasting & Revenue Leakage
        # This is a standalone add-on module - does NOT modify existing functionality
        render_spare_parts_planning_page()

    elif page_to_display == "🔍 Q&A System":
        st.header("Interactive Q&A System")

        uploaded_file = st.file_uploader("Upload your dataset (CSV):", type=["csv"])
        if uploaded_file is not None:
            try:
                st.session_state.uploaded_data = pd.read_csv(uploaded_file)
                st.session_state['data_columns'] = st.session_state.uploaded_data.columns.tolist()
                st.write("Dataset uploaded successfully!")
            except Exception as e:
                st.error(f"Error in loading dataset: {str(e)}")

        if st.session_state.uploaded_data is not None:
            data_analyzer = DataAnalyzer(st.session_state.uploaded_data, llm)
        else:
            data_analyzer = None

        with st.expander("Conversation History", expanded=True):
            if len(st.session_state.conversation) == 0:
                st.write("No conversation history yet.")
            else:
                for message in st.session_state.conversation:
                    if message['sender'] == 'user':
                        st.markdown(f"**You:** {message['text']}")
                    else:
                        st.markdown(f"**Assistant:** {message['text']}")

        with st.form(key='query_form'):
            user_input = st.text_input(
                "Ask a question about your dataset or any business-related query:",
                placeholder="Type your query here...",
                key='user_input' 
            )
            submit_button = st.form_submit_button(label='Send')

        if submit_button:
            if user_input:
                st.session_state.conversation.append({'sender': 'user', 'text': user_input})

                with st.spinner("Generating response..."):
                    if st.session_state.uploaded_data is not None:
                        data_analyzer = DataAnalyzer(st.session_state.uploaded_data, llm)
                        response_text = data_analyzer.process_question(user_input)
                    else:
                        conversation = [{'sender': 'user', 'text': user_input}]
                        response = llm.conversational_response(conversation)
                        response_text = response['text']

                st.session_state.conversation.append({'sender': 'assistant', 'text': response_text})
                st.success("Response received!")
                st.write(f"**Assistant:** {response_text}")

    elif page_to_display == "📊 Data Insights":
        st.header("Data Insights and Visualization")

        if st.session_state.uploaded_data is not None:
            data = st.session_state.uploaded_data
            data_analyzer = DataAnalyzer(data, llm)

            st.subheader("Dataset Overview")
            st.write(data.head())

            st.subheader("Numeric Column Statistics")
            numeric_stats = data.describe().transpose()
            st.table(numeric_stats)

            st.subheader("Automated Trend Analysis")
            trend_insights = data_analyzer.automated_trend_analysis()
            st.write(trend_insights)

            st.subheader("Time Series Analysis")
            data_analyzer.time_series_analysis()

            st.subheader("Interactive Data Visualization")
            columns = data.columns.tolist()
            x_axis = st.selectbox("Select X-axis", columns)
            y_axis = st.selectbox("Select Y-axis", columns)
            chart_type = st.selectbox("Select Chart Type", ["Line Chart", "Bar Chart", "Scatter Plot"])
            if st.button("Generate Chart"):
                fig = data_analyzer.generate_interactive_visualization(x_axis, y_axis, chart_type)
                st.plotly_chart(fig)

            st.subheader("Segment Analysis")
            segment_column = st.selectbox("Select Segment Column", columns)
            segment_insights = data_analyzer.segment_analysis(segment_column)
            st.write(segment_insights)

            st.subheader("Key Findings Summary")
            data_analyzer.key_findings_summary()
        else:
            st.write("Please upload a dataset in the Q&A System page to view insights.")

    elif page_to_display == "💡 AI-Powered Strategies":
        st.header("AI-Powered Business Strategy Suggestions")
        st.write("AI will suggest actionable strategies based on your uploaded data or company analysis.")

        if 'strategies' not in st.session_state:
            st.session_state['strategies'] = None
        if 'risk_analysis' not in st.session_state:
            st.session_state['risk_analysis'] = None
        if 'resource_estimates' not in st.session_state:
            st.session_state['resource_estimates'] = None

        if st.session_state.uploaded_data is not None:
            data = st.session_state.uploaded_data

            if st.button("Generate AI-Powered Strategies"):
                with st.spinner("Analyzing data and generating strategies..."):
                    
                    data_summary_df = data.describe(include='all').transpose().round(2)
                    data_summary = data_summary_df.to_string()

                    
                    max_length = 1500 
                    if len(data_summary) > max_length:
                        data_summary = data_summary[:max_length] + "\n... [Data truncated]"

                    strategies = llm.generate_strategic_recommendations(data_summary)
                    st.session_state['strategies'] = strategies
                    st.session_state['risk_analysis'] = None  
                    st.session_state['resource_estimates'] = None
                    st.write(f"**AI Strategy Suggestions:**\n{strategies}")

            
            if st.session_state['strategies']:
                strategies = st.session_state['strategies']

              
                if st.button("Generate Risk Analysis"):
                    with st.spinner("Performing risk analysis..."):
                        risk_analysis = llm.generate_risk_analysis(strategies)
                        st.session_state['risk_analysis'] = risk_analysis
                    st.write(f"**Risk Analysis:**\n{risk_analysis}")

                
                elif st.session_state['risk_analysis']:
                    st.write(f"**Risk Analysis:**\n{st.session_state['risk_analysis']}")

                
                if st.button("Estimate Resources"):
                    with st.spinner("Estimating resources..."):
                        resource_estimates = llm.estimate_resources(strategies)
                        st.session_state['resource_estimates'] = resource_estimates
                    st.write(f"**Resource Estimates:**\n{resource_estimates}")

                
                elif st.session_state['resource_estimates']:
                    st.write(f"**Resource Estimates:**\n{st.session_state['resource_estimates']}")

        elif st.session_state.financial_data is not None:
            company_data = st.session_state.financial_data

            if st.button("Generate AI-Powered Strategies"):
                with st.spinner("Analyzing financial data and generating strategies..."):
                    data_summary = company_data.to_string()
                    strategies = llm.generate_strategic_recommendations(data_summary)
                    st.session_state['strategies'] = strategies
                    st.session_state['risk_analysis'] = None  
                    st.session_state['resource_estimates'] = None
                st.write(f"**AI Strategy Suggestions:**\n{strategies}")

            if st.session_state['strategies']:
                strategies = st.session_state['strategies']

                
                if st.button("Generate Risk Analysis"):
                    with st.spinner("Performing risk analysis..."):
                        risk_analysis = llm.generate_risk_analysis(strategies)
                        st.session_state['risk_analysis'] = risk_analysis
                    st.write(f"**Risk Analysis:**\n{risk_analysis}")

                
                elif st.session_state['risk_analysis']:
                    st.write(f"**Risk Analysis:**\n{st.session_state['risk_analysis']}")

                
                if st.button("Estimate Resources"):
                    with st.spinner("Estimating resources..."):
                        resource_estimates = llm.estimate_resources(strategies)
                        st.session_state['resource_estimates'] = resource_estimates
                    st.write(f"**Resource Estimates:**\n{resource_estimates}")

                
                elif st.session_state['resource_estimates']:
                    st.write(f"**Resource Estimates:**\n{st.session_state['resource_estimates']}")

        else:
            st.write("Please upload a dataset or perform a company analysis first to generate strategies.")

        
        st.subheader("Simulate Your Custom Strategy")
        custom_strategy_input = st.text_area("Describe your custom strategy:")
        if st.button("Simulate Custom Strategy"):
            if custom_strategy_input:
                with st.spinner("Simulating strategy..."):
                    data_or_company = st.session_state.uploaded_data if st.session_state.uploaded_data is not None else st.session_state.financial_data
                    if data_or_company is not None:
                        data_summary_df = data_or_company.describe(include='all').transpose().round(2)
                        data_summary = data_summary_df.to_string()
                        max_length = 1500  
                        if len(data_summary) > max_length:
                            data_summary = data_summary[:max_length] + "\n... [Data truncated]"
                    else:
                        data_summary = "No data available."
                    simulation_result = llm.simulate_custom_strategy(custom_strategy_input, data_summary)
                    st.write(f"**Simulation Result:**\n{simulation_result}")
            else:
                st.write("Please enter a strategy to simulate.")

    display_footer()

if __name__ == "__main__":
    main()