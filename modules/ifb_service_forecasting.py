# modules/ifb_service_forecasting.py

"""
IFB Service Ecosystem Forecasting & Analytics System

This module provides comprehensive AI-driven forecasting and insights for IFB's
service ecosystem, including:
- 30/60/90-day demand forecasting
- Service volume predictions
- Spare parts usage and inventory planning
- Warranty claims forecasting
- Procurement optimization
- Revenue leakage identification
- Location-specific analytics (branches, regions, franchises)
- Franchise performance reporting
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class IFBServiceForecasting:
    """
    Comprehensive forecasting and analytics system for IFB's service ecosystem
    """

    def __init__(self, data, llm):
        """
        Initialize the forecasting system

        Args:
            data: DataFrame with service ecosystem data
            llm: LLM interface for AI-generated insights
        """
        self.data = data.copy()
        self.llm = llm
        self.models = {}
        self.forecasts = {}
        self.insights = {}
        self.prepare_data()

    def prepare_data(self):
        """Prepare and engineer features from the dataset"""
        # Convert date columns to datetime
        date_columns = self.data.select_dtypes(include=['object']).columns
        for col in date_columns:
            if 'date' in col.lower():
                try:
                    self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                except:
                    pass

        # Sort by date if date column exists
        date_col = self.get_date_column()
        if date_col:
            self.data = self.data.sort_values(date_col)

            # Extract temporal features
            self.data['Year'] = self.data[date_col].dt.year
            self.data['Month'] = self.data[date_col].dt.month
            self.data['Quarter'] = self.data[date_col].dt.quarter
            self.data['Day_of_Week'] = self.data[date_col].dt.dayofweek
            self.data['Week_of_Year'] = self.data[date_col].dt.isocalendar().week
            self.data['Is_Weekend'] = self.data['Day_of_Week'].isin([5, 6]).astype(int)

    def get_date_column(self):
        """Identify the date column in the dataset"""
        for col in self.data.columns:
            if 'date' in col.lower() and pd.api.types.is_datetime64_any_dtype(self.data[col]):
                return col
        return None

    def run_analysis(self):
        """Main method to run the complete analysis"""
        st.header("🔧 IFB Service Ecosystem - AI Forecasting & Analytics")
        st.markdown("""
        Comprehensive forecasting and insights for IFB's nationwide service network.
        Predict demand, optimize inventory, identify revenue leakages, and empower franchise partners.
        """)

        # Create main tabs
        tabs = st.tabs([
            "📊 Executive Dashboard",
            "📈 Demand Forecasting",
            "🔧 Service Volume Analysis",
            "📦 Spare Parts Planning",
            "🛡️ Warranty Claims",
            "🏢 Location Intelligence",
            "🤝 Franchise Performance",
            "💸 Revenue Optimization",
            "📋 Reports & Insights"
        ])

        with tabs[0]:
            self.executive_dashboard()

        with tabs[1]:
            self.demand_forecasting_hub()

        with tabs[2]:
            self.service_volume_analysis()
            # Add machine status and service attribute correlation
            self.service_attribute_correlation_analysis()

        with tabs[3]:
            self.spare_parts_planning()
            # Add product and location intelligence
            self.product_location_intelligence()

        with tabs[4]:
            self.warranty_claims_analysis()

        with tabs[5]:
            self.location_intelligence()

        with tabs[6]:
            self.franchise_performance()

        with tabs[7]:
            self.revenue_optimization()

        with tabs[8]:
            self.reports_and_insights()

    def executive_dashboard(self):
        """Executive-level dashboard with KPIs and trends"""
        st.subheader("📊 Executive Dashboard - Service Ecosystem Overview")

        # Calculate KPIs
        date_col = self.get_date_column()

        # Service volume metrics
        total_services = len(self.data) if 'Service_ID' in self.data.columns else len(self.data)

        # Revenue metrics
        total_revenue = self.data['Service_Revenue'].sum() if 'Service_Revenue' in self.data.columns else 0
        avg_service_value = self.data['Service_Revenue'].mean() if 'Service_Revenue' in self.data.columns else 0

        # Spare parts metrics
        total_parts_used = self.data['Parts_Used'].sum() if 'Parts_Used' in self.data.columns else 0
        parts_revenue = self.data['Parts_Revenue'].sum() if 'Parts_Revenue' in self.data.columns else 0

        # Warranty metrics
        warranty_claims = len(self.data[self.data['Warranty_Claim'] == 1]) if 'Warranty_Claim' in self.data.columns else 0
        warranty_cost = self.data[self.data['Warranty_Claim'] == 1]['Service_Cost'].sum() if 'Warranty_Claim' in self.data.columns and 'Service_Cost' in self.data.columns else 0

        # Display KPIs
        st.markdown("### 🎯 Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Service Calls", f"{total_services:,}")
            if 'Service_Revenue' in self.data.columns:
                st.metric("Service Revenue", f"₹{total_revenue:,.2f}")

        with col2:
            if 'Service_Revenue' in self.data.columns:
                st.metric("Avg Service Value", f"₹{avg_service_value:,.2f}")
            st.metric("Spare Parts Used", f"{total_parts_used:,}")

        with col3:
            st.metric("Parts Revenue", f"₹{parts_revenue:,.2f}")
            st.metric("Warranty Claims", f"{warranty_claims:,}")

        with col4:
            warranty_rate = (warranty_claims / total_services * 100) if total_services > 0 else 0
            st.metric("Warranty Claim Rate", f"{warranty_rate:.2f}%")
            st.metric("Warranty Cost", f"₹{warranty_cost:,.2f}")

        # Trend visualization
        if date_col:
            st.markdown("### 📈 Service Volume Trends")

            # Aggregate by month
            monthly_data = self.data.copy()
            monthly_data['Year_Month'] = monthly_data[date_col].dt.to_period('M')

            monthly_agg = monthly_data.groupby('Year_Month').agg({
                'Service_ID': 'count' if 'Service_ID' in monthly_data.columns else lambda x: len(x),
                'Service_Revenue': 'sum' if 'Service_Revenue' in monthly_data.columns else lambda x: 0,
                'Parts_Used': 'sum' if 'Parts_Used' in monthly_data.columns else lambda x: 0
            })
            monthly_agg.index = monthly_agg.index.to_timestamp()

            # Create dual-axis chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Scatter(x=monthly_agg.index, y=monthly_agg.iloc[:, 0],
                          name='Service Calls', line=dict(color='#4ecdc4', width=3)),
                secondary_y=False
            )

            if 'Service_Revenue' in self.data.columns:
                fig.add_trace(
                    go.Scatter(x=monthly_agg.index, y=monthly_agg['Service_Revenue'],
                              name='Revenue', line=dict(color='#ff6b6b', width=3)),
                    secondary_y=True
                )

            fig.update_xaxes(title_text="Month")
            fig.update_yaxes(title_text="Service Calls", secondary_y=False)
            fig.update_yaxes(title_text="Revenue (₹)", secondary_y=True)
            fig.update_layout(title="Service Calls & Revenue Trend", hovermode='x unified')

            st.plotly_chart(fig, use_container_width=True)

        # Location performance
        if 'Location' in self.data.columns or 'Branch' in self.data.columns:
            location_col = 'Location' if 'Location' in self.data.columns else 'Branch'

            st.markdown("### 🏢 Top Performing Locations")

            top_locations = self.data.groupby(location_col).agg({
                'Service_ID': 'count' if 'Service_ID' in self.data.columns else lambda x: len(x),
                'Service_Revenue': 'sum' if 'Service_Revenue' in self.data.columns else lambda x: 0
            }).nlargest(10, 'Service_ID' if 'Service_ID' in self.data.columns else self.data.columns[0])

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(top_locations.reset_index(), x=location_col,
                           y='Service_ID' if 'Service_ID' in top_locations.columns else top_locations.columns[0],
                           title="Top 10 Locations by Service Volume",
                           color='Service_ID' if 'Service_ID' in top_locations.columns else top_locations.columns[0],
                           color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'Service_Revenue' in top_locations.columns:
                    fig = px.pie(top_locations.reset_index(), values='Service_Revenue',
                               names=location_col,
                               title="Revenue Distribution - Top 10 Locations")
                    st.plotly_chart(fig, use_container_width=True)

        # AI-generated insights
        st.markdown("### 🤖 AI-Generated Executive Insights")

        exec_prompt = f"""
        Analyze this IFB service ecosystem data and provide executive-level insights:

        - Total Service Calls: {total_services:,}
        - Total Revenue: ₹{total_revenue:,.2f}
        - Average Service Value: ₹{avg_service_value:,.2f}
        - Spare Parts Used: {total_parts_used:,}
        - Parts Revenue: ₹{parts_revenue:,.2f}
        - Warranty Claims: {warranty_claims:,} ({warranty_rate:.2f}%)
        - Warranty Cost: ₹{warranty_cost:,.2f}

        Provide:
        1. Top 3 business insights
        2. Operational efficiency assessment
        3. Priority areas for improvement
        4. Growth opportunities
        """

        with st.spinner("Generating executive insights..."):
            insights = self.llm.conversational_response([{'sender': 'user', 'text': exec_prompt}])['text']
        st.write(insights)

    def demand_forecasting_hub(self):
        """Multi-period demand forecasting (30/60/90 days)"""
        st.subheader("📈 AI-Powered Demand Forecasting (30/60/90 Days)")

        date_col = self.get_date_column()
        if not date_col:
            st.warning("No date column found. Please ensure your dataset has a date column for forecasting.")
            return

        # Select forecasting target
        st.markdown("### 🎯 Select Forecasting Target")

        forecast_options = []
        if 'Service_ID' in self.data.columns:
            forecast_options.append("Service Volume")
        if 'Parts_Used' in self.data.columns:
            forecast_options.append("Spare Parts Demand")
        if 'Service_Revenue' in self.data.columns:
            forecast_options.append("Service Revenue")
        if 'Warranty_Claim' in self.data.columns:
            forecast_options.append("Warranty Claims")

        if not forecast_options:
            st.error("No forecasting targets available in the dataset")
            return

        target_type = st.selectbox("Choose what to forecast:", forecast_options)

        # Prepare time series data
        ts_data = self.data.copy()
        ts_data['Date'] = ts_data[date_col]
        ts_data = ts_data.sort_values('Date')

        # Aggregate by day
        if target_type == "Service Volume":
            daily_data = ts_data.groupby('Date').size().reset_index(name='Value')
            metric_name = "Service Calls"
        elif target_type == "Spare Parts Demand":
            daily_data = ts_data.groupby('Date')['Parts_Used'].sum().reset_index(name='Value')
            metric_name = "Parts Units"
        elif target_type == "Service Revenue":
            daily_data = ts_data.groupby('Date')['Service_Revenue'].sum().reset_index(name='Value')
            metric_name = "Revenue (₹)"
        else:  # Warranty Claims
            daily_data = ts_data.groupby('Date')['Warranty_Claim'].sum().reset_index(name='Value')
            metric_name = "Claims Count"

        if len(daily_data) < 30:
            st.warning(f"Insufficient data for forecasting. Need at least 30 days, found {len(daily_data)} days.")
            return

        # Feature engineering for ML model
        daily_data['Day_Num'] = range(len(daily_data))
        daily_data['Month'] = daily_data['Date'].dt.month
        daily_data['Day_of_Week'] = daily_data['Date'].dt.dayofweek
        daily_data['Quarter'] = daily_data['Date'].dt.quarter
        daily_data['Is_Weekend'] = daily_data['Day_of_Week'].isin([5, 6]).astype(int)

        # Add rolling averages
        daily_data['MA_7'] = daily_data['Value'].rolling(window=7, min_periods=1).mean()
        daily_data['MA_30'] = daily_data['Value'].rolling(window=30, min_periods=1).mean()

        # Train forecasting model
        features = ['Day_Num', 'Month', 'Day_of_Week', 'Quarter', 'Is_Weekend', 'MA_7', 'MA_30']
        X = daily_data[features].fillna(0)
        y = daily_data['Value']

        # Split data (use last 20% for validation)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        st.markdown("### 🤖 Training Forecasting Model...")

        model_choice = st.radio("Select Forecasting Model:",
                               ["Gradient Boosting (Recommended)", "Random Forest"])

        if model_choice == "Gradient Boosting (Recommended)":
            model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                             max_depth=5, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

        model.fit(X_train, y_train)

        # Validate model
        y_pred_test = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
        r2 = r2_score(y_test, y_pred_test)

        # Display model performance
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("MAPE", f"{mape:.2f}%")
        col4.metric("R² Score", f"{r2:.3f}")

        # Generate forecasts for 30, 60, 90 days
        st.markdown("### 🔮 Forecast Results")

        last_date = daily_data['Date'].max()
        last_day_num = daily_data['Day_Num'].max()

        # Create forecast dataframes
        forecast_30 = self.generate_forecast(last_date, last_day_num, 30, model, daily_data)
        forecast_60 = self.generate_forecast(last_date, last_day_num, 60, model, daily_data)
        forecast_90 = self.generate_forecast(last_date, last_day_num, 90, model, daily_data)

        # Visualize forecasts
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=daily_data['Date'], y=daily_data['Value'],
            mode='lines', name='Historical',
            line=dict(color='#4ecdc4', width=2)
        ))

        # 30-day forecast
        fig.add_trace(go.Scatter(
            x=forecast_30['Date'], y=forecast_30['Forecast'],
            mode='lines', name='30-Day Forecast',
            line=dict(color='#ffd93d', width=2, dash='dash')
        ))

        # 60-day forecast
        fig.add_trace(go.Scatter(
            x=forecast_60['Date'], y=forecast_60['Forecast'],
            mode='lines', name='60-Day Forecast',
            line=dict(color='#ff9a76', width=2, dash='dash')
        ))

        # 90-day forecast
        fig.add_trace(go.Scatter(
            x=forecast_90['Date'], y=forecast_90['Forecast'],
            mode='lines', name='90-Day Forecast',
            line=dict(color='#ff6b6b', width=2, dash='dash')
        ))

        fig.update_layout(
            title=f"{target_type} - Historical & Forecasted Trend",
            xaxis_title="Date",
            yaxis_title=metric_name,
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Forecast summaries
        st.markdown("### 📊 Forecast Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 30-Day Forecast")
            total_30 = forecast_30['Forecast'].sum()
            avg_30 = forecast_30['Forecast'].mean()
            st.metric("Total", f"{total_30:,.0f}")
            st.metric("Daily Average", f"{avg_30:,.0f}")

            # Growth vs historical
            hist_avg = daily_data['Value'].tail(30).mean()
            growth_30 = ((avg_30 - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0
            st.metric("vs Last 30 Days", f"{growth_30:+.1f}%")

        with col2:
            st.markdown("#### 60-Day Forecast")
            total_60 = forecast_60['Forecast'].sum()
            avg_60 = forecast_60['Forecast'].mean()
            st.metric("Total", f"{total_60:,.0f}")
            st.metric("Daily Average", f"{avg_60:,.0f}")

            growth_60 = ((avg_60 - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0
            st.metric("vs Last 30 Days", f"{growth_60:+.1f}%")

        with col3:
            st.markdown("#### 90-Day Forecast")
            total_90 = forecast_90['Forecast'].sum()
            avg_90 = forecast_90['Forecast'].mean()
            st.metric("Total", f"{total_90:,.0f}")
            st.metric("Daily Average", f"{avg_90:,.0f}")

            growth_90 = ((avg_90 - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0
            st.metric("vs Last 30 Days", f"{growth_90:+.1f}%")

        # Downloadable forecasts
        st.markdown("### 📥 Download Forecasts")

        # Combine all forecasts
        forecast_30['Period'] = '30-Day'
        forecast_60['Period'] = '60-Day'
        forecast_90['Period'] = '90-Day'

        all_forecasts = pd.concat([forecast_30, forecast_60, forecast_90])
        csv = all_forecasts.to_csv(index=False)

        st.download_button(
            label="Download All Forecasts (CSV)",
            data=csv,
            file_name=f"ifb_{target_type.lower().replace(' ', '_')}_forecast.csv",
            mime="text/csv"
        )

        # AI Insights on forecast
        st.markdown("### 🤖 AI-Generated Forecast Insights")

        forecast_prompt = f"""
        Analyze this {target_type} forecast for IFB's service ecosystem:

        Historical 30-day average: {hist_avg:,.0f} {metric_name}
        30-day forecast average: {avg_30:,.0f} ({growth_30:+.1f}% change)
        60-day forecast average: {avg_60:,.0f} ({growth_60:+.1f}% change)
        90-day forecast average: {avg_90:,.0f} ({growth_90:+.1f}% change)

        Model Performance:
        - MAE: {mae:.2f}
        - MAPE: {mape:.2f}%
        - R² Score: {r2:.3f}

        Provide:
        1. Interpretation of the forecast trend
        2. Business implications for IFB
        3. Recommended actions for operations team
        4. Resource planning suggestions
        5. Risk factors to monitor
        """

        with st.spinner("Generating forecast insights..."):
            insights = self.llm.conversational_response([{'sender': 'user', 'text': forecast_prompt}])['text']
        st.write(insights)

    def generate_forecast(self, last_date, last_day_num, days, model, historical_data):
        """Generate forecast for specified number of days"""
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)

        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Day_Num': range(last_day_num + 1, last_day_num + 1 + days),
            'Month': [d.month for d in future_dates],
            'Day_of_Week': [d.dayofweek for d in future_dates],
            'Quarter': [d.quarter for d in future_dates],
            'Is_Weekend': [1 if d.dayofweek in [5, 6] else 0 for d in future_dates]
        })

        # Use last known moving averages
        last_ma_7 = historical_data['MA_7'].iloc[-1]
        last_ma_30 = historical_data['MA_30'].iloc[-1]

        forecast_df['MA_7'] = last_ma_7
        forecast_df['MA_30'] = last_ma_30

        # Generate predictions
        features = ['Day_Num', 'Month', 'Day_of_Week', 'Quarter', 'Is_Weekend', 'MA_7', 'MA_30']
        forecast_df['Forecast'] = model.predict(forecast_df[features])

        # Ensure non-negative forecasts
        forecast_df['Forecast'] = forecast_df['Forecast'].clip(lower=0)

        return forecast_df[['Date', 'Forecast']]

    def service_volume_analysis(self):
        """Analyze service call patterns and trends"""
        st.subheader("🔧 Service Volume Analysis")

        # Service type distribution
        if 'Service_Type' in self.data.columns:
            st.markdown("### Service Type Distribution")

            service_counts = self.data['Service_Type'].value_counts()

            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(values=service_counts.values, names=service_counts.index,
                           title="Service Types Distribution")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(x=service_counts.index, y=service_counts.values,
                           title="Service Volume by Type",
                           labels={'x': 'Service Type', 'y': 'Count'},
                           color=service_counts.values,
                           color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)

        # Service completion time analysis
        if 'Service_Duration' in self.data.columns:
            st.markdown("### Service Duration Analysis")

            avg_duration = self.data['Service_Duration'].mean()
            median_duration = self.data['Service_Duration'].median()

            col1, col2, col3 = st.columns(3)
            col1.metric("Average Duration", f"{avg_duration:.1f} hours")
            col2.metric("Median Duration", f"{median_duration:.1f} hours")
            col3.metric("Max Duration", f"{self.data['Service_Duration'].max():.1f} hours")

            fig = px.histogram(self.data, x='Service_Duration', nbins=30,
                             title="Service Duration Distribution",
                             labels={'Service_Duration': 'Duration (hours)'})
            st.plotly_chart(fig, use_container_width=True)

        # Technician performance
        if 'Technician_ID' in self.data.columns:
            st.markdown("### Technician Performance")

            tech_performance = self.data.groupby('Technician_ID').agg({
                'Service_ID': 'count' if 'Service_ID' in self.data.columns else lambda x: len(x),
                'Service_Revenue': 'sum' if 'Service_Revenue' in self.data.columns else lambda x: 0,
                'Service_Duration': 'mean' if 'Service_Duration' in self.data.columns else lambda x: 0
            }).nlargest(15, 'Service_ID' if 'Service_ID' in self.data.columns else tech_performance.columns[0])

            st.dataframe(tech_performance.style.format({
                col: '{:,.0f}' for col in tech_performance.columns if 'ID' in col
            }).format({
                col: '₹{:,.2f}' for col in tech_performance.columns if 'Revenue' in col
            }).format({
                col: '{:.1f} hrs' for col in tech_performance.columns if 'Duration' in col
            }))

    def spare_parts_planning(self):
        """Spare parts demand analysis and inventory optimization"""
        st.subheader("📦 Spare Parts Planning & Inventory Optimization")

        if 'Part_ID' not in self.data.columns and 'Part_Name' not in self.data.columns:
            st.warning("No spare parts data available in the dataset")
            return

        part_col = 'Part_Name' if 'Part_Name' in self.data.columns else 'Part_ID'

        # Parts usage analysis
        st.markdown("### 🔍 Parts Demand Analysis")

        parts_usage = self.data.groupby(part_col).agg({
            'Parts_Used': 'sum' if 'Parts_Used' in self.data.columns else lambda x: len(x),
            'Parts_Cost': 'sum' if 'Parts_Cost' in self.data.columns else lambda x: 0
        }).sort_values('Parts_Used' if 'Parts_Used' in self.data.columns else parts_usage.columns[0],
                       ascending=False)

        # Top parts by usage
        st.markdown("#### Top 20 Parts by Usage")
        top_parts = parts_usage.head(20)

        fig = px.bar(top_parts.reset_index(), x=part_col,
                    y='Parts_Used' if 'Parts_Used' in top_parts.columns else top_parts.columns[0],
                    title="Most Used Spare Parts",
                    color='Parts_Used' if 'Parts_Used' in top_parts.columns else top_parts.columns[0],
                    color_continuous_scale='Reds')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        # ABC Analysis for inventory management
        st.markdown("### 📊 ABC Analysis for Inventory Management")

        # Calculate cumulative percentage
        parts_sorted = parts_usage.copy()
        parts_sorted['Cumulative_Usage'] = parts_sorted['Parts_Used' if 'Parts_Used' in parts_sorted.columns else parts_sorted.columns[0]].cumsum()
        total_usage = parts_sorted['Parts_Used' if 'Parts_Used' in parts_sorted.columns else parts_sorted.columns[0]].sum()
        parts_sorted['Cumulative_Pct'] = (parts_sorted['Cumulative_Usage'] / total_usage * 100)

        # Classify into ABC categories
        parts_sorted['Category'] = 'C'
        parts_sorted.loc[parts_sorted['Cumulative_Pct'] <= 80, 'Category'] = 'A'
        parts_sorted.loc[(parts_sorted['Cumulative_Pct'] > 80) & (parts_sorted['Cumulative_Pct'] <= 95), 'Category'] = 'B'

        category_counts = parts_sorted['Category'].value_counts()

        col1, col2, col3 = st.columns(3)
        col1.metric("Category A (Critical)", f"{category_counts.get('A', 0)} parts",
                   help="Top 80% of usage - Keep high stock")
        col2.metric("Category B (Important)", f"{category_counts.get('B', 0)} parts",
                   help="15% of usage - Moderate stock")
        col3.metric("Category C (Regular)", f"{category_counts.get('C', 0)} parts",
                   help="5% of usage - Low stock")

        # Display parts by category
        category_filter = st.selectbox("View parts in category:", ['All', 'A', 'B', 'C'])

        if category_filter != 'All':
            display_parts = parts_sorted[parts_sorted['Category'] == category_filter]
        else:
            display_parts = parts_sorted

        st.dataframe(display_parts.reset_index().head(50))

        # Location-wise parts demand
        if 'Location' in self.data.columns or 'Branch' in self.data.columns:
            location_col = 'Location' if 'Location' in self.data.columns else 'Branch'

            st.markdown("### 🏢 Location-Wise Parts Distribution Strategy")

            location_parts = self.data.groupby([location_col, part_col]).agg({
                'Parts_Used': 'sum' if 'Parts_Used' in self.data.columns else lambda x: len(x)
            }).reset_index()

            # Pivot for heatmap
            pivot_data = location_parts.pivot_table(
                index=part_col,
                columns=location_col,
                values='Parts_Used' if 'Parts_Used' in location_parts.columns else location_parts.columns[2],
                fill_value=0
            )

            # Show top 20 parts and top 10 locations
            if len(pivot_data) > 20:
                pivot_data = pivot_data.iloc[:20]
            if len(pivot_data.columns) > 10:
                pivot_data = pivot_data.iloc[:, :10]

            fig = px.imshow(pivot_data,
                          labels=dict(x="Location", y="Part", color="Usage"),
                          title="Parts Demand Heatmap by Location",
                          aspect="auto",
                          color_continuous_scale='RdYlGn_r')
            fig.update_xaxes(side="bottom")
            st.plotly_chart(fig, use_container_width=True)

        # Calculate EOQ and Safety Stock for top parts
        st.markdown("### 📊 Inventory Optimization - EOQ & Safety Stock")

        # Calculate for top 10 Category A parts
        top_parts_for_eoq = parts_sorted[parts_sorted['Category'] == 'A'].head(10)

        if not top_parts_for_eoq.empty:
            eoq_results = []
            for idx, row in top_parts_for_eoq.iterrows():
                annual_demand = row['Parts_Used' if 'Parts_Used' in row.index else row.index[0]] * 12  # Annualized

                # Assumptions for EOQ calculation (can be customized)
                ordering_cost = 500  # ₹ per order
                holding_cost_rate = 0.25  # 25% of part cost per year
                avg_part_cost = row.get('Parts_Cost', 100) / max(row.get('Parts_Used', 1), 1) if 'Parts_Cost' in row.index else 100

                # EOQ Formula: sqrt((2 * D * S) / H)
                # D = annual demand, S = ordering cost, H = holding cost per unit
                holding_cost_per_unit = avg_part_cost * holding_cost_rate

                if holding_cost_per_unit > 0:
                    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
                else:
                    eoq = 0

                # Safety stock calculation (for 95% service level, Z = 1.65)
                # Assuming 7-day lead time and demand variability
                daily_demand = annual_demand / 365
                demand_std_dev = daily_demand * 0.3  # Assume 30% coefficient of variation
                lead_time_days = 7
                safety_stock = 1.65 * demand_std_dev * np.sqrt(lead_time_days)

                # Reorder point = (demand during lead time) + safety stock
                reorder_point = (daily_demand * lead_time_days) + safety_stock

                eoq_results.append({
                    'Part': idx,
                    'Annual_Demand': int(annual_demand),
                    'EOQ': int(eoq),
                    'Safety_Stock': int(safety_stock),
                    'Reorder_Point': int(reorder_point),
                    'Avg_Cost': f"₹{avg_part_cost:,.0f}"
                })

            eoq_df = pd.DataFrame(eoq_results)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Economic Order Quantity (EOQ)")
                st.info("""
                **EOQ Definition:** Optimal order quantity that minimizes total inventory costs (ordering + holding).

                **How to use:** Order this quantity each time stock hits the reorder point.

                **Formula:** EOQ = √((2 × Annual Demand × Ordering Cost) / Holding Cost per Unit)
                """)

            with col2:
                st.markdown("#### Safety Stock & Reorder Point")
                st.info("""
                **Safety Stock:** Buffer inventory to prevent stockouts during demand variability or lead time delays.

                **Reorder Point:** Stock level that triggers a new order.

                **Formula:** Reorder Point = (Daily Demand × Lead Time) + Safety Stock
                """)

            st.dataframe(eoq_df.style.format({
                'Annual_Demand': '{:,}',
                'EOQ': '{:,}',
                'Safety_Stock': '{:,}',
                'Reorder_Point': '{:,}'
            }), use_container_width=True)

            st.markdown("#### 💡 Interpretation")
            st.success(f"""
            **For Category A critical parts:**
            - **Average EOQ:** {eoq_df['EOQ'].mean():,.0f} units per order
            - **Average Safety Stock:** {eoq_df['Safety_Stock'].mean():,.0f} units as buffer
            - **Action:** When stock drops to reorder point, place order for EOQ quantity
            - **Benefit:** Optimizes ordering frequency and reduces inventory carrying costs
            """)

        # Procurement recommendations
        st.markdown("### 💡 AI-Powered Procurement Recommendations")

        procurement_prompt = f"""
        Based on this spare parts usage data for IFB service ecosystem:

        Total unique parts: {len(parts_usage)}
        Category A (Critical) parts: {category_counts.get('A', 0)}
        Category B (Important) parts: {category_counts.get('B', 0)}
        Category C (Regular) parts: {category_counts.get('C', 0)}

        Top 5 most used parts:
        {parts_usage.head(5).to_string()}

        Provide:
        1. Inventory stocking strategy for each ABC category
        2. Vendor management recommendations for Category A parts
        3. Cost reduction opportunities through consolidation
        4. Distribution strategy across service locations
        5. Risk mitigation for supply chain disruptions
        """

        with st.spinner("Generating procurement insights..."):
            insights = self.llm.conversational_response([{'sender': 'user', 'text': procurement_prompt}])['text']
        st.write(insights)

    def service_attribute_correlation_analysis(self):
        """Analyze correlation between failures, machine status, and service attributes"""
        st.markdown("### 🔍 Advanced Correlation Analysis")
        st.markdown("""
        **This section identifies:**
        - **Where products are used** (machine status, location patterns)
        - **Why failures occur** (correlation with service types, warranty status)
        - **Whether service coverage explains behavior** (AMC vs EW vs LMC patterns)
        """)

        # Machine Status Analysis
        if 'Machine_Status' in self.data.columns or any('machine' in col.lower() for col in self.data.columns):
            machine_col = 'Machine_Status' if 'Machine_Status' in self.data.columns else [col for col in self.data.columns if 'machine' in col.lower()][0]

            st.markdown("#### 🖥️ Machine Status vs Service Demand")

            machine_analysis = self.data.groupby(machine_col).agg({
                'Service_ID': 'count' if 'Service_ID' in self.data.columns else lambda x: len(x),
                'Parts_Used': 'sum' if 'Parts_Used' in self.data.columns else lambda x: 0,
                'Service_Revenue': 'sum' if 'Service_Revenue' in self.data.columns else lambda x: 0,
                'Warranty_Claim': 'sum' if 'Warranty_Claim' in self.data.columns else lambda x: 0
            }).reset_index()

            machine_analysis.columns = [machine_col, 'Service_Calls', 'Parts_Used', 'Revenue', 'Warranty_Claims']

            if len(machine_analysis) > 0:
                machine_analysis['Avg_Parts_per_Service'] = machine_analysis['Parts_Used'] / machine_analysis['Service_Calls']
                machine_analysis['Warranty_Rate'] = (machine_analysis['Warranty_Claims'] / machine_analysis['Service_Calls'] * 100)

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.bar(machine_analysis, x=machine_col, y='Service_Calls',
                               title='Service Volume by Machine Status',
                               color='Warranty_Rate',
                               color_continuous_scale='Reds',
                               labels={machine_col: 'Machine Status', 'Service_Calls': 'Service Calls'})
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.scatter(machine_analysis, x='Service_Calls', y='Avg_Parts_per_Service',
                                   size='Revenue', color=machine_col,
                                   title='Service Intensity vs Parts Consumption',
                                   labels={'Avg_Parts_per_Service': 'Avg Parts per Service'})
                    st.plotly_chart(fig, use_container_width=True)

                st.dataframe(machine_analysis.style.format({
                    'Service_Calls': '{:,}',
                    'Parts_Used': '{:,}',
                    'Revenue': '₹{:,.2f}',
                    'Warranty_Claims': '{:,}',
                    'Avg_Parts_per_Service': '{:.2f}',
                    'Warranty_Rate': '{:.2f}%'
                }), use_container_width=True)

                # AI Insights on machine status correlation
                st.markdown("#### 🤖 AI Insights - Machine Status Impact")
                worst_status = machine_analysis.nlargest(1, 'Avg_Parts_per_Service').iloc[0]

                machine_prompt = f"""
                Analyze this machine status vs service demand correlation for IFB:

                Machine Status with highest parts consumption: {worst_status[machine_col]}
                - Service Calls: {worst_status['Service_Calls']:,}
                - Average Parts per Service: {worst_status['Avg_Parts_per_Service']:.2f}
                - Warranty Claim Rate: {worst_status['Warranty_Rate']:.2f}%
                - Total Revenue: ₹{worst_status['Revenue']:,.2f}

                Full breakdown:
                {machine_analysis.to_string()}

                Provide insights on:
                1. What does this machine status pattern reveal about product usage and failure modes?
                2. Why are certain machine statuses associated with higher parts consumption?
                3. Recommended actions to reduce service costs for high-consumption statuses
                4. Preventive maintenance strategies based on these patterns
                """

                with st.spinner("Analyzing machine status patterns..."):
                    insights = self.llm.conversational_response([{'sender': 'user', 'text': machine_prompt}])['text']
                st.write(insights)

        # Service Type Analysis (AMC, EW, LMC, etc.)
        if 'Service_Type' in self.data.columns:
            st.markdown("#### 🛡️ Service Coverage Type Analysis")

            service_type_analysis = self.data.groupby('Service_Type').agg({
                'Service_ID': 'count' if 'Service_ID' in self.data.columns else lambda x: len(x),
                'Parts_Used': 'sum' if 'Parts_Used' in self.data.columns else lambda x: 0,
                'Service_Revenue': 'sum' if 'Service_Revenue' in self.data.columns else lambda x: 0,
                'Parts_Revenue': 'sum' if 'Parts_Revenue' in self.data.columns else lambda x: 0,
                'Warranty_Claim': 'sum' if 'Warranty_Claim' in self.data.columns else lambda x: 0
            }).reset_index()

            service_type_analysis.columns = ['Service_Type', 'Service_Calls', 'Parts_Used', 'Service_Revenue', 'Parts_Revenue', 'Warranty_Claims']

            if len(service_type_analysis) > 0:
                service_type_analysis['Warranty_Rate'] = (service_type_analysis['Warranty_Claims'] / service_type_analysis['Service_Calls'] * 100)
                service_type_analysis['Parts_Revenue_Ratio'] = (service_type_analysis['Parts_Revenue'] / service_type_analysis['Service_Revenue'] * 100)

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.pie(service_type_analysis, values='Service_Calls', names='Service_Type',
                               title='Service Call Distribution by Type')
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.bar(service_type_analysis, x='Service_Type', y=['Service_Revenue', 'Parts_Revenue'],
                               title='Revenue Breakdown by Service Type',
                               labels={'value': 'Revenue (₹)', 'variable': 'Revenue Type'},
                               barmode='group')
                    st.plotly_chart(fig, use_container_width=True)

                st.dataframe(service_type_analysis.style.format({
                    'Service_Calls': '{:,}',
                    'Parts_Used': '{:,}',
                    'Service_Revenue': '₹{:,.2f}',
                    'Parts_Revenue': '₹{:,.2f}',
                    'Warranty_Claims': '{:,}',
                    'Warranty_Rate': '{:.2f}%',
                    'Parts_Revenue_Ratio': '{:.2f}%'
                }), use_container_width=True)

                st.markdown("#### 📊 Service Type Insights")

                # Identify patterns
                amc_data = service_type_analysis[service_type_analysis['Service_Type'].str.contains('AMC', case=False, na=False)]
                ew_data = service_type_analysis[service_type_analysis['Service_Type'].str.contains('EW|EXTENDED', case=False, na=False)]

                insights = []
                if not amc_data.empty:
                    amc_warranty_rate = amc_data['Warranty_Rate'].mean()
                    insights.append(f"**AMC Services:** {amc_warranty_rate:.1f}% warranty claim rate - {'HIGH (requires investigation)' if amc_warranty_rate > 15 else 'Normal'}")

                if not ew_data.empty:
                    ew_parts_ratio = ew_data['Parts_Revenue_Ratio'].mean()
                    insights.append(f"**Extended Warranty:** {ew_parts_ratio:.1f}% parts revenue ratio - {'Profitable' if ew_parts_ratio < 40 else 'Review pricing'}")

                if insights:
                    for insight in insights:
                        st.info(insight)

                # AI-generated service type recommendations
                service_type_prompt = f"""
                Analyze service type performance for IFB service ecosystem:

                {service_type_analysis.to_string()}

                Provide:
                1. Which service types (AMC/EW/LMC) are most profitable and why?
                2. Are warranty claim patterns indicating coverage abuse or genuine quality issues?
                3. Recommended pricing adjustments for each service type
                4. Cross-sell opportunities (e.g., converting pay-per-service to AMC)
                5. Service type portfolio optimization strategy
                """

                st.markdown("#### 🤖 AI Service Coverage Recommendations")
                with st.spinner("Analyzing service type patterns..."):
                    service_insights = self.llm.conversational_response([{'sender': 'user', 'text': service_type_prompt}])['text']
                st.write(service_insights)

        st.markdown("---")

    def product_location_intelligence(self):
        """Product and location intelligence with delivery delay analysis"""
        st.markdown("### 📍 Product & Location Intelligence")
        st.markdown("""
        **This section identifies:**
        - **Most ordered products** and their demand patterns
        - **High-demand locations** and regional trends
        - **Delivery delays** and their impact on operations
        - **Location-specific patterns** in product consumption
        """)

        date_col = self.get_date_column()

        # Most ordered products
        if 'Product_Category' in self.data.columns:
            st.markdown("#### 🏆 Most Ordered Products")

            product_demand = self.data.groupby('Product_Category').agg({
                'Service_ID': 'count' if 'Service_ID' in self.data.columns else lambda x: len(x),
                'Parts_Used': 'sum' if 'Parts_Used' in self.data.columns else lambda x: 0,
                'Service_Revenue': 'sum' if 'Service_Revenue' in self.data.columns else lambda x: 0
            }).reset_index()

            product_demand.columns = ['Product', 'Service_Calls', 'Parts_Used', 'Revenue']
            product_demand = product_demand.sort_values('Service_Calls', ascending=False).head(10)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(product_demand, x='Product', y='Service_Calls',
                           title='Top 10 Products by Service Demand',
                           color='Service_Calls',
                           color_continuous_scale='Blues')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.treemap(product_demand, path=['Product'], values='Revenue',
                               title='Revenue Distribution by Product')
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(product_demand.style.format({
                'Service_Calls': '{:,}',
                'Parts_Used': '{:,}',
                'Revenue': '₹{:,.2f}'
            }).background_gradient(cmap='Blues', subset=['Service_Calls']), use_container_width=True)

        # High-demand locations
        if 'Location' in self.data.columns or 'Branch' in self.data.columns:
            location_col = 'Location' if 'Location' in self.data.columns else 'Branch'

            st.markdown("#### 🌍 High-Demand Locations")

            location_demand = self.data.groupby(location_col).agg({
                'Service_ID': 'count' if 'Service_ID' in self.data.columns else lambda x: len(x),
                'Service_Revenue': 'sum' if 'Service_Revenue' in self.data.columns else lambda x: 0,
                'Parts_Used': 'sum' if 'Parts_Used' in self.data.columns else lambda x: 0
            }).reset_index()

            location_demand.columns = ['Location', 'Service_Calls', 'Revenue', 'Parts_Used']
            location_demand['Revenue_per_Call'] = location_demand['Revenue'] / location_demand['Service_Calls']
            location_demand = location_demand.sort_values('Service_Calls', ascending=False).head(15)

            fig = px.scatter(location_demand, x='Service_Calls', y='Revenue_per_Call',
                           size='Parts_Used', color='Location',
                           title='Location Performance Matrix',
                           labels={
                               'Service_Calls': 'Service Volume',
                               'Revenue_per_Call': 'Revenue per Service Call (₹)',
                               'Parts_Used': 'Parts Consumption'
                           },
                           hover_data=['Location', 'Service_Calls', 'Revenue'])
            st.plotly_chart(fig, use_container_width=True)

        # Delivery delay analysis
        if date_col and 'Service_Date' in self.data.columns:
            # Check for ETA or delivery date columns
            eta_col = None
            for col in self.data.columns:
                if 'eta' in col.lower() or 'delivery' in col.lower() or 'completion' in col.lower():
                    eta_col = col
                    break

            if eta_col:
                st.markdown("#### ⏱️ Delivery Delay Analysis")

                # Calculate delays
                delay_data = self.data.copy()
                delay_data['Service_Date_dt'] = pd.to_datetime(delay_data['Service_Date'], errors='coerce')
                delay_data['ETA_dt'] = pd.to_datetime(delay_data[eta_col], errors='coerce')
                delay_data['Delay_Days'] = (delay_data['Service_Date_dt'] - delay_data['ETA_dt']).dt.days

                # Filter valid delays
                delay_data = delay_data[delay_data['Delay_Days'].notna()]

                if len(delay_data) > 0:
                    # Overall delay metrics
                    col1, col2, col3, col4 = st.columns(4)

                    delayed_services = len(delay_data[delay_data['Delay_Days'] > 0])
                    on_time_services = len(delay_data[delay_data['Delay_Days'] <= 0])
                    avg_delay = delay_data[delay_data['Delay_Days'] > 0]['Delay_Days'].mean() if delayed_services > 0 else 0
                    max_delay = delay_data['Delay_Days'].max()

                    col1.metric("Total Services", f"{len(delay_data):,}")
                    col2.metric("Delayed Services", f"{delayed_services:,}", delta=f"{(delayed_services/len(delay_data)*100):.1f}%", delta_color="inverse")
                    col3.metric("On-Time Services", f"{on_time_services:,}", delta=f"{(on_time_services/len(delay_data)*100):.1f}%")
                    col4.metric("Avg Delay", f"{avg_delay:.1f} days")

                    # Products affected by delays
                    if 'Product_Category' in delay_data.columns:
                        st.markdown("##### 📦 Products Most Affected by Delays")

                        product_delays = delay_data[delay_data['Delay_Days'] > 0].groupby('Product_Category').agg({
                            'Service_ID': 'count' if 'Service_ID' in delay_data.columns else lambda x: len(x),
                            'Delay_Days': 'mean'
                        }).reset_index()

                        product_delays.columns = ['Product', 'Delayed_Services', 'Avg_Delay_Days']
                        product_delays = product_delays.sort_values('Delayed_Services', ascending=False).head(10)

                        fig = px.bar(product_delays, x='Product', y='Delayed_Services',
                                   color='Avg_Delay_Days',
                                   title='Top 10 Products by Delay Frequency',
                                   color_continuous_scale='Reds',
                                   labels={'Delayed_Services': 'Number of Delayed Services'})
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)

                    # Location-specific delays
                    if location_col in delay_data.columns:
                        st.markdown("##### 🌍 Location-Specific Delay Patterns")

                        location_delays = delay_data[delay_data['Delay_Days'] > 0].groupby(location_col).agg({
                            'Service_ID': 'count' if 'Service_ID' in delay_data.columns else lambda x: len(x),
                            'Delay_Days': ['mean', 'max']
                        }).reset_index()

                        location_delays.columns = ['Location', 'Delayed_Count', 'Avg_Delay', 'Max_Delay']
                        location_delays['On_Time_Rate'] = 100 - (location_delays['Delayed_Count'] / delay_data.groupby(location_col).size().values * 100)
                        location_delays = location_delays.sort_values('Delayed_Count', ascending=False).head(10)

                        col1, col2 = st.columns(2)

                        with col1:
                            fig = px.bar(location_delays, x='Location', y='Delayed_Count',
                                       title='Locations with Most Delays',
                                       color='Avg_Delay',
                                       color_continuous_scale='Oranges')
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.dataframe(location_delays.style.format({
                                'Delayed_Count': '{:,}',
                                'Avg_Delay': '{:.1f} days',
                                'Max_Delay': '{:.0f} days',
                                'On_Time_Rate': '{:.1f}%'
                            }).background_gradient(cmap='Reds', subset=['Avg_Delay']), use_container_width=True)

                    # AI Insights on delays
                    st.markdown("#### 🤖 AI Delay Impact Analysis")

                    delay_prompt = f"""
                    Analyze delivery delay patterns for IFB service operations:

                    Total Services: {len(delay_data):,}
                    Delayed Services: {delayed_services:,} ({(delayed_services/len(delay_data)*100):.1f}%)
                    Average Delay: {avg_delay:.1f} days
                    Maximum Delay: {max_delay:.0f} days

                    Products most affected:
                    {product_delays.to_string() if 'Product_Category' in delay_data.columns else 'Data not available'}

                    Provide insights on:
                    1. What are the business implications of these delay patterns?
                    2. Which products or locations should be prioritized for improvement?
                    3. Root causes of delays (supply chain, logistics, inventory management)?
                    4. Recommended actions to reduce delays by 50% in next 90 days
                    5. Impact on customer satisfaction and retention
                    """

                    with st.spinner("Analyzing delay patterns..."):
                        delay_insights = self.llm.conversational_response([{'sender': 'user', 'text': delay_prompt}])['text']
                    st.write(delay_insights)

                else:
                    st.info("No delay data available for analysis")
            else:
                st.info("No delivery/ETA date column found for delay analysis")

        st.markdown("---")

    def warranty_claims_analysis(self):
        """Warranty claims analysis and forecasting"""
        st.subheader("🛡️ Warranty Claims Analysis & Prediction")

        if 'Warranty_Claim' not in self.data.columns:
            st.warning("No warranty claim data available in the dataset")
            return

        # Warranty metrics
        total_claims = self.data['Warranty_Claim'].sum()
        total_services = len(self.data)
        claim_rate = (total_claims / total_services * 100) if total_services > 0 else 0

        warranty_cost = self.data[self.data['Warranty_Claim'] == 1]['Service_Cost'].sum() if 'Service_Cost' in self.data.columns else 0
        avg_claim_cost = warranty_cost / total_claims if total_claims > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Claims", f"{total_claims:,}")
        col2.metric("Claim Rate", f"{claim_rate:.2f}%")
        col3.metric("Total Warranty Cost", f"₹{warranty_cost:,.2f}")
        col4.metric("Avg Cost per Claim", f"₹{avg_claim_cost:,.2f}")

        # Claims by product category
        if 'Product_Category' in self.data.columns:
            st.markdown("### Claims by Product Category")

            category_claims = self.data.groupby('Product_Category').agg({
                'Warranty_Claim': 'sum',
                'Service_ID': 'count' if 'Service_ID' in self.data.columns else lambda x: len(x)
            })
            category_claims['Claim_Rate'] = (category_claims['Warranty_Claim'] / category_claims.iloc[:, 1] * 100)
            category_claims = category_claims.sort_values('Warranty_Claim', ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(category_claims.reset_index(), x='Product_Category', y='Warranty_Claim',
                           title="Warranty Claims by Category",
                           color='Claim_Rate',
                           color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(category_claims.style.format({
                    'Warranty_Claim': '{:,.0f}',
                    'Claim_Rate': '{:.2f}%'
                }))

        # Time-based analysis
        date_col = self.get_date_column()
        if date_col:
            st.markdown("### Warranty Claims Trend")

            monthly_claims = self.data.copy()
            monthly_claims['Year_Month'] = monthly_claims[date_col].dt.to_period('M')

            claims_trend = monthly_claims.groupby('Year_Month').agg({
                'Warranty_Claim': 'sum',
                'Service_ID': 'count' if 'Service_ID' in monthly_claims.columns else lambda x: len(x)
            })
            claims_trend.index = claims_trend.index.to_timestamp()
            claims_trend['Claim_Rate'] = (claims_trend['Warranty_Claim'] / claims_trend.iloc[:, 1] * 100)

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Bar(x=claims_trend.index, y=claims_trend['Warranty_Claim'],
                      name='Claims Count', marker_color='#ff6b6b'),
                secondary_y=False
            )

            fig.add_trace(
                go.Scatter(x=claims_trend.index, y=claims_trend['Claim_Rate'],
                          name='Claim Rate (%)', line=dict(color='#4ecdc4', width=3)),
                secondary_y=True
            )

            fig.update_xaxes(title_text="Month")
            fig.update_yaxes(title_text="Claims Count", secondary_y=False)
            fig.update_yaxes(title_text="Claim Rate (%)", secondary_y=True)
            fig.update_layout(title="Warranty Claims Trend", hovermode='x unified')

            st.plotly_chart(fig, use_container_width=True)

    def location_intelligence(self):
        """Location-specific analytics and insights"""
        st.subheader("🏢 Location Intelligence & Branch Performance")

        if 'Location' not in self.data.columns and 'Branch' not in self.data.columns:
            st.warning("No location data available in the dataset")
            return

        location_col = 'Location' if 'Location' in self.data.columns else 'Branch'

        # Location performance metrics
        st.markdown("### 📍 Location Performance Overview")

        location_metrics = self.data.groupby(location_col).agg({
            'Service_ID': 'count' if 'Service_ID' in self.data.columns else lambda x: len(x),
            'Service_Revenue': 'sum' if 'Service_Revenue' in self.data.columns else lambda x: 0,
            'Parts_Used': 'sum' if 'Parts_Used' in self.data.columns else lambda x: 0,
            'Warranty_Claim': 'sum' if 'Warranty_Claim' in self.data.columns else lambda x: 0
        })

        if 'Service_Revenue' in self.data.columns:
            location_metrics['Avg_Service_Value'] = location_metrics['Service_Revenue'] / location_metrics.iloc[:, 0]

        location_metrics = location_metrics.sort_values(
            'Service_ID' if 'Service_ID' in location_metrics.columns else location_metrics.columns[0],
            ascending=False
        )

        # Display top and bottom performers
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🌟 Top 10 Performing Locations")
            top_locations = location_metrics.head(10)
            st.dataframe(top_locations.style.format({
                col: '{:,.0f}' for col in top_locations.columns if 'ID' in col or 'Used' in col or 'Claim' in col
            }).format({
                col: '₹{:,.2f}' for col in top_locations.columns if 'Revenue' in col or 'Value' in col
            }).background_gradient(cmap='Greens', subset=[top_locations.columns[0]]))

        with col2:
            st.markdown("#### ⚠️ Bottom 10 Locations (Need Attention)")
            bottom_locations = location_metrics.tail(10)
            st.dataframe(bottom_locations.style.format({
                col: '{:,.0f}' for col in bottom_locations.columns if 'ID' in col or 'Used' in col or 'Claim' in col
            }).format({
                col: '₹{:,.2f}' for col in bottom_locations.columns if 'Revenue' in col or 'Value' in col
            }).background_gradient(cmap='Reds', subset=[bottom_locations.columns[0]]))

        # Regional comparison
        if 'Region' in self.data.columns:
            st.markdown("### 🗺️ Regional Performance Comparison")

            regional_metrics = self.data.groupby('Region').agg({
                'Service_ID': 'count' if 'Service_ID' in self.data.columns else lambda x: len(x),
                'Service_Revenue': 'sum' if 'Service_Revenue' in self.data.columns else lambda x: 0,
                location_col: 'nunique'
            })

            regional_metrics.columns = ['Service_Calls', 'Revenue', 'Locations']
            regional_metrics['Avg_per_Location'] = regional_metrics['Service_Calls'] / regional_metrics['Locations']

            fig = px.bar(regional_metrics.reset_index(), x='Region', y='Service_Calls',
                        title="Service Volume by Region",
                        color='Avg_per_Location',
                        color_continuous_scale='Blues',
                        hover_data=['Revenue', 'Locations'])
            st.plotly_chart(fig, use_container_width=True)

        # Location-specific recommendations
        st.markdown("### 💡 Location Optimization Recommendations")

        selected_location = st.selectbox(f"Select {location_col} for detailed analysis:",
                                        location_metrics.index.tolist())

        if selected_location:
            location_data = self.data[self.data[location_col] == selected_location]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Service Calls", f"{len(location_data):,}")
                if 'Service_Revenue' in location_data.columns:
                    st.metric("Total Revenue", f"₹{location_data['Service_Revenue'].sum():,.2f}")

            with col2:
                if 'Parts_Used' in location_data.columns:
                    st.metric("Parts Used", f"{location_data['Parts_Used'].sum():,}")
                if 'Warranty_Claim' in location_data.columns:
                    warranty_rate = (location_data['Warranty_Claim'].sum() / len(location_data) * 100)
                    st.metric("Warranty Claim Rate", f"{warranty_rate:.2f}%")

            with col3:
                if 'Technician_ID' in location_data.columns:
                    st.metric("Active Technicians", f"{location_data['Technician_ID'].nunique()}")
                if 'Service_Duration' in location_data.columns:
                    st.metric("Avg Service Time", f"{location_data['Service_Duration'].mean():.1f} hrs")

    def franchise_performance(self):
        """Franchise partner performance tracking and reporting"""
        st.subheader("🤝 Franchise Performance Dashboard")

        if 'Franchise_ID' not in self.data.columns and 'Partner_ID' not in self.data.columns:
            st.info("This analysis works best with franchise/partner data. Adapting to available location data...")
            franchise_col = 'Location' if 'Location' in self.data.columns else 'Branch' if 'Branch' in self.data.columns else None

            if not franchise_col:
                st.warning("No franchise or location data available")
                return
        else:
            franchise_col = 'Franchise_ID' if 'Franchise_ID' in self.data.columns else 'Partner_ID'

        st.markdown("### 📊 Franchise Performance Scorecard")

        # Calculate comprehensive metrics for each franchise
        franchise_metrics = self.data.groupby(franchise_col).agg({
            'Service_ID': 'count' if 'Service_ID' in self.data.columns else lambda x: len(x),
            'Service_Revenue': 'sum' if 'Service_Revenue' in self.data.columns else lambda x: 0,
            'Parts_Revenue': 'sum' if 'Parts_Revenue' in self.data.columns else lambda x: 0,
            'Warranty_Claim': 'sum' if 'Warranty_Claim' in self.data.columns else lambda x: 0,
            'Customer_Satisfaction': 'mean' if 'Customer_Satisfaction' in self.data.columns else lambda x: 0
        })

        # Calculate derived metrics
        franchise_metrics['Total_Revenue'] = franchise_metrics.get('Service_Revenue', 0) + franchise_metrics.get('Parts_Revenue', 0)
        franchise_metrics['Warranty_Rate'] = (franchise_metrics['Warranty_Claim'] / franchise_metrics.iloc[:, 0] * 100)

        # Performance ranking
        franchise_metrics['Performance_Score'] = (
            franchise_metrics.iloc[:, 0].rank(pct=True) * 0.4 +  # Service volume
            franchise_metrics['Total_Revenue'].rank(pct=True) * 0.3 +  # Revenue
            (100 - franchise_metrics['Warranty_Rate']).rank(pct=True) * 0.3  # Quality (lower warranty rate)
        ) * 100

        franchise_metrics = franchise_metrics.sort_values('Performance_Score', ascending=False)

        # Display performance tiers
        st.markdown("#### Performance Tiers")

        col1, col2, col3 = st.columns(3)

        platinum = franchise_metrics[franchise_metrics['Performance_Score'] >= 80]
        gold = franchise_metrics[(franchise_metrics['Performance_Score'] >= 60) & (franchise_metrics['Performance_Score'] < 80)]
        silver = franchise_metrics[franchise_metrics['Performance_Score'] < 60]

        with col1:
            st.metric("🏆 Platinum Partners", len(platinum), help="Performance Score ≥ 80")
        with col2:
            st.metric("🥇 Gold Partners", len(gold), help="Performance Score 60-79")
        with col3:
            st.metric("🥈 Silver Partners", len(silver), help="Performance Score < 60")

        # Detailed franchise table
        st.markdown("### 📋 Detailed Franchise Performance")

        display_metrics = franchise_metrics.copy()
        display_metrics['Tier'] = 'Silver'
        display_metrics.loc[display_metrics['Performance_Score'] >= 60, 'Tier'] = 'Gold'
        display_metrics.loc[display_metrics['Performance_Score'] >= 80, 'Tier'] = 'Platinum'

        st.dataframe(display_metrics.style.format({
            col: '{:,.0f}' for col in display_metrics.columns if 'ID' in col or 'Claim' in col
        }).format({
            col: '₹{:,.2f}' for col in display_metrics.columns if 'Revenue' in col
        }).format({
            'Warranty_Rate': '{:.2f}%',
            'Performance_Score': '{:.1f}',
            'Customer_Satisfaction': '{:.2f}'
        }).background_gradient(cmap='RdYlGn', subset=['Performance_Score']))

        # Growth opportunity analysis
        st.markdown("### 🚀 Growth Opportunity Analysis")

        # Select a franchise for detailed analysis
        selected_franchise = st.selectbox(f"Select {franchise_col} for detailed insights:",
                                         franchise_metrics.index.tolist())

        if selected_franchise:
            franchise_data = self.data[self.data[franchise_col] == selected_franchise]
            franchise_stats = franchise_metrics.loc[selected_franchise]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"#### Performance Summary: {selected_franchise}")
                st.metric("Performance Score", f"{franchise_stats['Performance_Score']:.1f}/100")
                st.metric("Tier", display_metrics.loc[selected_franchise, 'Tier'])
                st.metric("Service Calls", f"{franchise_stats.iloc[0]:,.0f}")
                st.metric("Total Revenue", f"₹{franchise_stats['Total_Revenue']:,.2f}")

            with col2:
                st.markdown("#### vs Network Average")
                network_avg_revenue = franchise_metrics['Total_Revenue'].mean()
                revenue_vs_avg = ((franchise_stats['Total_Revenue'] - network_avg_revenue) / network_avg_revenue * 100)

                network_avg_calls = franchise_metrics.iloc[:, 0].mean()
                calls_vs_avg = ((franchise_stats.iloc[0] - network_avg_calls) / network_avg_calls * 100)

                st.metric("Revenue vs Average", f"{revenue_vs_avg:+.1f}%")
                st.metric("Service Calls vs Average", f"{calls_vs_avg:+.1f}%")
                st.metric("Warranty Rate", f"{franchise_stats['Warranty_Rate']:.2f}%")

                if 'Customer_Satisfaction' in franchise_stats.index:
                    st.metric("Customer Satisfaction", f"{franchise_stats['Customer_Satisfaction']:.2f}/5.0")

            # AI-generated franchise insights
            st.markdown("### 🤖 AI-Generated Partner Insights")

            franchise_prompt = f"""
            Analyze this franchise partner's performance for IFB:

            Franchise: {selected_franchise}
            Performance Score: {franchise_stats['Performance_Score']:.1f}/100
            Tier: {display_metrics.loc[selected_franchise, 'Tier']}
            Service Calls: {franchise_stats.iloc[0]:,.0f}
            Total Revenue: ₹{franchise_stats['Total_Revenue']:,.2f}
            Revenue vs Network Avg: {revenue_vs_avg:+.1f}%
            Warranty Claim Rate: {franchise_stats['Warranty_Rate']:.2f}%

            Provide:
            1. Performance assessment (strengths and areas for improvement)
            2. Specific growth opportunities
            3. Actionable recommendations to improve performance
            4. Support or training needs
            5. Potential to move to next tier
            """

            with st.spinner("Generating franchise insights..."):
                insights = self.llm.conversational_response([{'sender': 'user', 'text': franchise_prompt}])['text']
            st.write(insights)

            # Download franchise report
            st.markdown("### 📥 Generate Franchise Report")

            if st.button("Generate Detailed Report"):
                report_data = {
                    'Metric': ['Service Calls', 'Total Revenue', 'Service Revenue', 'Parts Revenue',
                              'Warranty Claims', 'Warranty Rate', 'Performance Score', 'Tier'],
                    'Value': [
                        f"{franchise_stats.iloc[0]:,.0f}",
                        f"₹{franchise_stats['Total_Revenue']:,.2f}",
                        f"₹{franchise_stats.get('Service_Revenue', 0):,.2f}",
                        f"₹{franchise_stats.get('Parts_Revenue', 0):,.2f}",
                        f"{franchise_stats['Warranty_Claim']:,.0f}",
                        f"{franchise_stats['Warranty_Rate']:.2f}%",
                        f"{franchise_stats['Performance_Score']:.1f}",
                        display_metrics.loc[selected_franchise, 'Tier']
                    ]
                }

                report_df = pd.DataFrame(report_data)
                csv = report_df.to_csv(index=False)

                st.download_button(
                    label=f"Download {selected_franchise} Report (CSV)",
                    data=csv,
                    file_name=f"ifb_franchise_report_{selected_franchise}.csv",
                    mime="text/csv"
                )

    def revenue_optimization(self):
        """Revenue leakage identification and optimization recommendations"""
        st.subheader("💸 Revenue Optimization & Leakage Analysis")

        st.markdown("""
        Identify revenue leakages specific to service operations and get actionable
        recommendations to maximize revenue and profitability.
        """)

        # Revenue leakage sources
        st.markdown("### 🔍 Revenue Leakage Sources")

        leakage_analysis = {}

        # 1. Warranty claim costs
        if 'Warranty_Claim' in self.data.columns and 'Service_Cost' in self.data.columns:
            warranty_cost = self.data[self.data['Warranty_Claim'] == 1]['Service_Cost'].sum()
            leakage_analysis['Warranty Costs'] = warranty_cost

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Revenue Lost to Warranty", f"₹{warranty_cost:,.2f}")
            with col2:
                avg_warranty_cost = warranty_cost / self.data['Warranty_Claim'].sum() if self.data['Warranty_Claim'].sum() > 0 else 0
                st.metric("Avg Cost per Warranty Claim", f"₹{avg_warranty_cost:,.2f}")

        # 2. Service duration inefficiencies
        if 'Service_Duration' in self.data.columns and 'Service_Revenue' in self.data.columns:
            # Calculate revenue per hour
            self.data['Revenue_per_Hour'] = self.data['Service_Revenue'] / self.data['Service_Duration']

            avg_rph = self.data['Revenue_per_Hour'].median()
            inefficient_services = self.data[self.data['Revenue_per_Hour'] < avg_rph * 0.7]  # <70% of median

            potential_loss = (avg_rph - inefficient_services['Revenue_per_Hour']).sum() * inefficient_services['Service_Duration'].sum()
            leakage_analysis['Service Inefficiency'] = abs(potential_loss)

            st.markdown("#### Service Time Efficiency")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Inefficient Service Calls", f"{len(inefficient_services):,}")
            with col2:
                st.metric("Potential Revenue Loss", f"₹{abs(potential_loss):,.2f}")

        # 3. Parts pricing optimization
        if 'Parts_Cost' in self.data.columns and 'Parts_Revenue' in self.data.columns:
            parts_margin = ((self.data['Parts_Revenue'] - self.data['Parts_Cost']) / self.data['Parts_Revenue'] * 100).mean()

            # Identify low-margin transactions
            self.data['Parts_Margin'] = ((self.data['Parts_Revenue'] - self.data['Parts_Cost']) / self.data['Parts_Revenue'] * 100)
            low_margin = self.data[self.data['Parts_Margin'] < 20]  # <20% margin

            margin_opportunity = low_margin['Parts_Revenue'].sum() * 0.1  # 10% improvement potential
            leakage_analysis['Parts Pricing'] = margin_opportunity

            st.markdown("#### Parts Pricing Optimization")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Parts Margin", f"{parts_margin:.2f}%")
                st.metric("Low Margin Transactions", f"{len(low_margin):,}")
            with col2:
                st.metric("Margin Improvement Potential", f"₹{margin_opportunity:,.2f}")

        # Total leakage summary
        st.markdown("### 💰 Total Revenue Leakage Summary")

        total_leakage = sum(leakage_analysis.values())
        total_revenue = self.data['Service_Revenue'].sum() if 'Service_Revenue' in self.data.columns else 0
        leakage_pct = (total_leakage / total_revenue * 100) if total_revenue > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Revenue Leakage", f"₹{total_leakage:,.2f}")
        col2.metric("% of Total Revenue", f"{leakage_pct:.2f}%")
        col3.metric("Recovery Potential (70%)", f"₹{total_leakage * 0.7:,.2f}")

        # Leakage breakdown
        if leakage_analysis:
            fig = px.pie(names=list(leakage_analysis.keys()),
                        values=list(leakage_analysis.values()),
                        title="Revenue Leakage Breakdown by Source")
            st.plotly_chart(fig, use_container_width=True)

        # AI recommendations
        st.markdown("### 🤖 AI-Powered Revenue Optimization Recommendations")

        optimization_prompt = f"""
        Analyze this revenue leakage data for IFB's service ecosystem:

        Total Revenue Leakage: ₹{total_leakage:,.2f} ({leakage_pct:.2f}% of revenue)

        Leakage Sources:
        {chr(10).join([f"- {k}: ₹{v:,.2f}" for k, v in leakage_analysis.items()])}

        Recovery Potential: ₹{total_leakage * 0.7:,.2f}

        Provide:
        1. Immediate actions (0-30 days) to stop revenue leakage
        2. Process improvements for service efficiency
        3. Pricing optimization strategies for parts
        4. Warranty cost reduction recommendations
        5. Expected impact of implementing recommendations
        6. Key metrics to track improvement
        """

        with st.spinner("Generating optimization recommendations..."):
            insights = self.llm.conversational_response([{'sender': 'user', 'text': optimization_prompt}])['text']
        st.write(insights)

    def reports_and_insights(self):
        """Generate comprehensive reports and actionable insights"""
        st.subheader("📋 Reports & Actionable Insights")

        st.markdown("""
        Generate comprehensive reports for stakeholders, franchise partners, and operations teams.
        """)

        # Report type selection
        report_type = st.selectbox("Select Report Type:", [
            "Executive Summary Report",
            "Operational Performance Report",
            "Franchise Partner Report",
            "Inventory Planning Report",
            "Revenue Analysis Report",
            "Custom Report"
        ])

        if report_type == "Executive Summary Report":
            self.generate_executive_report()
        elif report_type == "Operational Performance Report":
            self.generate_operational_report()
        elif report_type == "Franchise Partner Report":
            self.generate_franchise_report()
        elif report_type == "Inventory Planning Report":
            self.generate_inventory_report()
        elif report_type == "Revenue Analysis Report":
            self.generate_revenue_report()
        else:
            st.info("Custom report builder coming soon...")

    def generate_executive_report(self):
        """Generate executive summary report"""
        st.markdown("### 📊 Executive Summary Report")

        # Compile key metrics
        report_data = {
            'Metric': [],
            'Value': [],
            'Period': []
        }

        total_services = len(self.data)
        total_revenue = self.data['Service_Revenue'].sum() if 'Service_Revenue' in self.data.columns else 0

        report_data['Metric'].extend(['Total Service Calls', 'Total Revenue'])
        report_data['Value'].extend([f"{total_services:,}", f"₹{total_revenue:,.2f}"])
        report_data['Period'].extend(['Overall', 'Overall'])

        # Add more metrics...

        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df)

        # Download option
        csv = report_df.to_csv(index=False)
        st.download_button(
            label="Download Executive Report (CSV)",
            data=csv,
            file_name="ifb_executive_summary.csv",
            mime="text/csv"
        )

    def generate_operational_report(self):
        """Generate operational performance report"""
        st.markdown("### ⚙️ Operational Performance Report")
        st.info("Generating comprehensive operational metrics...")

    def generate_franchise_report(self):
        """Generate franchise partner report"""
        st.markdown("### 🤝 Franchise Partner Report")
        st.info("Select franchise partner for detailed report generation...")

    def generate_inventory_report(self):
        """Generate inventory planning report"""
        st.markdown("### 📦 Inventory Planning Report")
        st.info("Generating inventory and procurement recommendations...")

    def generate_revenue_report(self):
        """Generate revenue analysis report"""
        st.markdown("### 💰 Revenue Analysis Report")
        st.info("Generating detailed revenue analysis...")