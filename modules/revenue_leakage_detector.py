# modules/revenue_leakage_detector.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class RevenueLeakageDetector:
    """
    AI-driven Revenue Leakage Detection and Forecasting System

    This class identifies various types of revenue leakages including:
    - Excessive discounting patterns
    - Profit margin erosion
    - Underperforming products/categories
    - Regional performance issues
    - Customer profitability problems
    - Pricing inefficiencies
    """

    def __init__(self, data, llm):
        self.data = data.copy()
        self.llm = llm
        self.leakage_report = {}
        self.prepare_data()

    def prepare_data(self):
        """Prepare and clean the dataset for analysis"""
        # Convert date columns to datetime
        date_columns = self.data.select_dtypes(include=['object']).columns
        for col in date_columns:
            if 'date' in col.lower():
                try:
                    self.data[col] = pd.to_datetime(self.data[col], format='%d-%m-%Y', errors='coerce')
                except:
                    try:
                        self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                    except:
                        pass

        # Calculate additional metrics if not present
        if 'Sales' in self.data.columns and 'Profit' in self.data.columns:
            self.data['Profit_Margin'] = (self.data['Profit'] / self.data['Sales'] * 100).round(2)
            self.data['Revenue_Lost_to_Discount'] = (self.data['Sales'] * self.data.get('Discount', 0))
            self.data['Potential_Revenue'] = self.data['Sales'] / (1 - self.data.get('Discount', 0))

    def detect_leakages(self):
        """Main method to detect all types of revenue leakages"""
        st.header("🔍 AI-Powered Revenue Leakage Analysis")
        st.markdown("""
        This comprehensive analysis identifies revenue leakages across multiple dimensions:
        discounting, profitability, product performance, regional trends, and more.
        """)

        # Create tabs for different analyses
        tabs = st.tabs([
            "📊 Executive Summary",
            "💰 Discount Analysis",
            "📉 Profit Erosion",
            "📦 Product Performance",
            "🌍 Regional Analysis",
            "🔮 Forecasting",
            "📈 Anomaly Detection",
            "📋 Recommendations"
        ])

        with tabs[0]:
            self.executive_summary()

        with tabs[1]:
            self.analyze_discount_leakage()

        with tabs[2]:
            self.analyze_profit_erosion()

        with tabs[3]:
            self.analyze_product_performance()

        with tabs[4]:
            self.analyze_regional_performance()

        with tabs[5]:
            self.forecast_revenue_leakage()

        with tabs[6]:
            self.detect_anomalies()

        with tabs[7]:
            self.generate_recommendations()

    def executive_summary(self):
        """Generate executive summary of revenue leakages"""
        st.subheader("📊 Executive Summary - Revenue Leakage Overview")

        # Calculate key metrics
        total_sales = self.data['Sales'].sum() if 'Sales' in self.data.columns else 0
        total_profit = self.data['Profit'].sum() if 'Profit' in self.data.columns else 0
        avg_discount = self.data['Discount'].mean() * 100 if 'Discount' in self.data.columns else 0
        avg_profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0

        # Revenue lost to discounts
        revenue_lost_discount = self.data['Revenue_Lost_to_Discount'].sum() if 'Revenue_Lost_to_Discount' in self.data.columns else 0

        # Negative profit transactions
        negative_profit_count = len(self.data[self.data['Profit'] < 0]) if 'Profit' in self.data.columns else 0
        negative_profit_amount = self.data[self.data['Profit'] < 0]['Profit'].sum() if 'Profit' in self.data.columns else 0

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Sales", f"${total_sales:,.2f}")
            st.metric("Total Profit", f"${total_profit:,.2f}")

        with col2:
            st.metric("Avg Profit Margin", f"{avg_profit_margin:.2f}%")
            st.metric("Avg Discount", f"{avg_discount:.2f}%")

        with col3:
            st.metric("Revenue Lost to Discounts", f"${revenue_lost_discount:,.2f}")
            st.metric("Negative Profit Transactions", f"{negative_profit_count}")

        with col4:
            leakage_amount = revenue_lost_discount + abs(negative_profit_amount)
            st.metric("Total Revenue Leakage", f"${leakage_amount:,.2f}",
                     delta=f"-{(leakage_amount/total_sales*100):.2f}%", delta_color="inverse")
            potential_savings = leakage_amount * 0.7  # Assume 70% recoverable
            st.metric("Potential Savings", f"${potential_savings:,.2f}")

        # Visualization: Leakage breakdown
        st.subheader("Revenue Leakage Breakdown")

        leakage_data = pd.DataFrame({
            'Category': ['Discount Leakage', 'Negative Profit', 'Potential Savings'],
            'Amount': [revenue_lost_discount, abs(negative_profit_amount), potential_savings]
        })

        fig = px.bar(leakage_data, x='Category', y='Amount',
                     title='Revenue Leakage Components',
                     color='Category',
                     color_discrete_map={
                         'Discount Leakage': '#ff6b6b',
                         'Negative Profit': '#ee5a6f',
                         'Potential Savings': '#4ecdc4'
                     })
        st.plotly_chart(fig, use_container_width=True)

        # AI-generated insights
        st.subheader("🤖 AI-Generated Executive Insights")
        summary_prompt = f"""
        Analyze this business revenue leakage summary and provide executive-level insights:

        - Total Sales: ${total_sales:,.2f}
        - Total Profit: ${total_profit:,.2f}
        - Average Profit Margin: {avg_profit_margin:.2f}%
        - Average Discount: {avg_discount:.2f}%
        - Revenue Lost to Discounts: ${revenue_lost_discount:,.2f}
        - Negative Profit Transactions: {negative_profit_count}
        - Total Negative Profit Amount: ${negative_profit_amount:,.2f}
        - Total Revenue Leakage: ${leakage_amount:,.2f}

        Provide:
        1. Top 3 critical findings
        2. Impact assessment
        3. Priority areas for immediate action
        """

        with st.spinner("Generating AI insights..."):
            insights = self.llm.conversational_response([{'sender': 'user', 'text': summary_prompt}])['text']
        st.write(insights)

    def analyze_discount_leakage(self):
        """Analyze revenue leakage due to discounting patterns"""
        st.subheader("💰 Discount-Driven Revenue Leakage Analysis")

        if 'Discount' not in self.data.columns:
            st.warning("No discount data available in the dataset")
            return

        # Discount distribution
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Discount Distribution")
            fig = px.histogram(self.data, x='Discount', nbins=50,
                             title='Distribution of Discount Rates',
                             labels={'Discount': 'Discount Rate', 'count': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### High Discount Transactions")
            high_discount = self.data[self.data['Discount'] > 0.2]  # >20% discount
            high_discount_loss = high_discount['Revenue_Lost_to_Discount'].sum()

            st.metric("Transactions with >20% Discount", len(high_discount))
            st.metric("Revenue Lost (>20% discount)", f"${high_discount_loss:,.2f}")

            # Discount by category
            if 'Category' in self.data.columns:
                discount_by_cat = self.data.groupby('Category').agg({
                    'Discount': 'mean',
                    'Revenue_Lost_to_Discount': 'sum'
                }).sort_values('Revenue_Lost_to_Discount', ascending=False)
                st.dataframe(discount_by_cat.style.format({
                    'Discount': '{:.2%}',
                    'Revenue_Lost_to_Discount': '${:,.2f}'
                }))

        # Discount vs Profit analysis
        st.markdown("#### Discount Impact on Profitability")

        if 'Profit_Margin' in self.data.columns:
            fig = px.scatter(self.data.sample(min(1000, len(self.data))),
                           x='Discount', y='Profit_Margin',
                           color='Category' if 'Category' in self.data.columns else None,
                           title='Discount Rate vs Profit Margin',
                           labels={'Discount': 'Discount Rate', 'Profit_Margin': 'Profit Margin (%)'})
            st.plotly_chart(fig, use_container_width=True)

        # AI Insights
        st.markdown("#### 🤖 AI-Generated Discount Insights")
        discount_stats = self.data['Discount'].describe()
        discount_prompt = f"""
        Analyze this discount pattern data:

        {discount_stats.to_string()}

        High discount transactions (>20%): {len(high_discount)}
        Revenue lost to high discounts: ${high_discount_loss:,.2f}

        Provide insights on:
        1. Are discounts excessive or justified?
        2. Which product categories have problematic discount patterns?
        3. Specific recommendations to optimize discount strategy
        """

        with st.spinner("Analyzing discount patterns..."):
            insights = self.llm.conversational_response([{'sender': 'user', 'text': discount_prompt}])['text']
        st.write(insights)

    def analyze_profit_erosion(self):
        """Analyze profit margin erosion over time"""
        st.subheader("📉 Profit Margin Erosion Analysis")

        if 'Profit_Margin' not in self.data.columns:
            st.warning("Unable to calculate profit margins")
            return

        # Identify date column
        date_col = None
        for col in self.data.columns:
            if 'date' in col.lower() and pd.api.types.is_datetime64_any_dtype(self.data[col]):
                date_col = col
                break

        if date_col:
            # Time series analysis of profit margin
            time_data = self.data.sort_values(date_col)
            time_data['Year_Month'] = time_data[date_col].dt.to_period('M')

            monthly_metrics = time_data.groupby('Year_Month').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Profit_Margin': 'mean'
            })
            monthly_metrics.index = monthly_metrics.index.to_timestamp()

            # Plot profit margin trend
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_metrics.index, y=monthly_metrics['Profit_Margin'],
                                   mode='lines+markers', name='Avg Profit Margin',
                                   line=dict(color='#4ecdc4', width=3)))
            fig.update_layout(title='Profit Margin Trend Over Time',
                            xaxis_title='Date',
                            yaxis_title='Profit Margin (%)')
            st.plotly_chart(fig, use_container_width=True)

            # Calculate trend
            monthly_metrics['Period_Num'] = range(len(monthly_metrics))
            if len(monthly_metrics) > 2:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    monthly_metrics['Period_Num'], monthly_metrics['Profit_Margin']
                )

                trend_direction = "increasing" if slope > 0 else "decreasing"
                st.info(f"📊 Profit margin trend: **{trend_direction}** at {abs(slope):.2f}% per month")

        # Profit margin distribution
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Profit Margin Distribution")
            fig = px.histogram(self.data, x='Profit_Margin', nbins=50,
                             title='Distribution of Profit Margins')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Low/Negative Margin Analysis")
            low_margin = self.data[self.data['Profit_Margin'] < 5]  # <5% margin
            negative_margin = self.data[self.data['Profit_Margin'] < 0]

            st.metric("Transactions with <5% Margin", len(low_margin))
            st.metric("Negative Margin Transactions", len(negative_margin))
            st.metric("Total Loss (Negative Margin)",
                     f"${negative_margin['Profit'].sum():,.2f}" if len(negative_margin) > 0 else "$0.00")

        # Category-wise profit analysis
        if 'Category' in self.data.columns:
            st.markdown("#### Category-wise Profit Performance")
            cat_profit = self.data.groupby('Category').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Profit_Margin': 'mean'
            }).sort_values('Profit_Margin')

            fig = px.bar(cat_profit.reset_index(), x='Category', y='Profit_Margin',
                        title='Average Profit Margin by Category',
                        color='Profit_Margin',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(cat_profit.style.format({
                'Sales': '${:,.2f}',
                'Profit': '${:,.2f}',
                'Profit_Margin': '{:.2f}%'
            }))

    def analyze_product_performance(self):
        """Analyze underperforming products and categories"""
        st.subheader("📦 Product & Category Performance Analysis")

        if 'Category' in self.data.columns and 'Sub Category' in self.data.columns:
            # Category performance
            cat_performance = self.data.groupby('Category').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Discount': 'mean'
            }).sort_values('Profit')

            cat_performance['Profit_Margin'] = (cat_performance['Profit'] / cat_performance['Sales'] * 100)

            # Visualize category performance
            fig = px.scatter(cat_performance.reset_index(),
                           x='Sales', y='Profit',
                           size='Discount',
                           color='Profit_Margin',
                           hover_data=['Category'],
                           title='Category Performance: Sales vs Profit',
                           color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)

            # Top and bottom performers
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🌟 Top Performing Categories")
                top_cats = cat_performance.nlargest(5, 'Profit')
                st.dataframe(top_cats.style.format({
                    'Sales': '${:,.2f}',
                    'Profit': '${:,.2f}',
                    'Discount': '{:.2%}',
                    'Profit_Margin': '{:.2f}%'
                }).background_gradient(cmap='Greens', subset=['Profit']))

            with col2:
                st.markdown("#### ⚠️ Bottom Performing Categories")
                bottom_cats = cat_performance.nsmallest(5, 'Profit')
                st.dataframe(bottom_cats.style.format({
                    'Sales': '${:,.2f}',
                    'Profit': '${:,.2f}',
                    'Discount': '{:.2%}',
                    'Profit_Margin': '{:.2f}%'
                }).background_gradient(cmap='Reds', subset=['Profit']))

            # Sub-category analysis
            st.markdown("#### Sub-Category Deep Dive")
            subcat_performance = self.data.groupby('Sub Category').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Discount': 'mean'
            }).sort_values('Profit')

            subcat_performance['Profit_Margin'] = (subcat_performance['Profit'] / subcat_performance['Sales'] * 100)

            # Filter options
            filter_option = st.radio("Show:", ["All", "Profitable Only", "Loss-Making Only"])

            if filter_option == "Profitable Only":
                subcat_display = subcat_performance[subcat_performance['Profit'] > 0]
            elif filter_option == "Loss-Making Only":
                subcat_display = subcat_performance[subcat_performance['Profit'] < 0]
            else:
                subcat_display = subcat_performance

            st.dataframe(subcat_display.style.format({
                'Sales': '${:,.2f}',
                'Profit': '${:,.2f}',
                'Discount': '{:.2%}',
                'Profit_Margin': '{:.2f}%'
            }).background_gradient(cmap='RdYlGn', subset=['Profit_Margin']))

    def analyze_regional_performance(self):
        """Analyze regional revenue leakages"""
        st.subheader("🌍 Regional Performance Analysis")

        if 'Region' not in self.data.columns and 'State' not in self.data.columns:
            st.warning("No regional data available")
            return

        region_col = 'Region' if 'Region' in self.data.columns else 'State'

        # Regional performance metrics
        regional_metrics = self.data.groupby(region_col).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Discount': 'mean'
        })

        regional_metrics['Profit_Margin'] = (regional_metrics['Profit'] / regional_metrics['Sales'] * 100)
        regional_metrics = regional_metrics.sort_values('Profit', ascending=False)

        # Visualize regional performance
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(regional_metrics.reset_index(),
                        x=region_col, y='Sales',
                        title=f'Sales by {region_col}',
                        color='Profit_Margin',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(regional_metrics.reset_index(),
                        values='Profit', names=region_col,
                        title=f'Profit Distribution by {region_col}')
            st.plotly_chart(fig, use_container_width=True)

        # Regional leakage analysis
        st.markdown(f"#### {region_col} Performance Table")
        st.dataframe(regional_metrics.style.format({
            'Sales': '${:,.2f}',
            'Profit': '${:,.2f}',
            'Discount': '{:.2%}',
            'Profit_Margin': '{:.2f}%'
        }).background_gradient(cmap='RdYlGn', subset=['Profit_Margin']))

        # Identify problem regions
        problem_regions = regional_metrics[regional_metrics['Profit_Margin'] < regional_metrics['Profit_Margin'].mean()]

        if len(problem_regions) > 0:
            st.warning(f"⚠️ {len(problem_regions)} regions performing below average profit margin")
            st.dataframe(problem_regions.style.format({
                'Sales': '${:,.2f}',
                'Profit': '${:,.2f}',
                'Discount': '{:.2%}',
                'Profit_Margin': '{:.2f}%'
            }))

    def forecast_revenue_leakage(self):
        """Forecast future revenue leakage using ML models"""
        st.subheader("🔮 Revenue Leakage Forecasting")

        # Check for time-based data
        date_col = None
        for col in self.data.columns:
            if 'date' in col.lower() and pd.api.types.is_datetime64_any_dtype(self.data[col]):
                date_col = col
                break

        if not date_col:
            st.warning("No date column found for time series forecasting")
            return

        # Prepare time series data
        ts_data = self.data.sort_values(date_col).copy()
        ts_data['Year_Month'] = ts_data[date_col].dt.to_period('M')

        monthly_leakage = ts_data.groupby('Year_Month').agg({
            'Revenue_Lost_to_Discount': 'sum',
            'Profit': lambda x: abs(x[x < 0].sum()) if (x < 0).any() else 0,
            'Sales': 'sum'
        })

        monthly_leakage.columns = ['Discount_Leakage', 'Negative_Profit_Leakage', 'Sales']
        monthly_leakage['Total_Leakage'] = monthly_leakage['Discount_Leakage'] + monthly_leakage['Negative_Profit_Leakage']
        monthly_leakage['Leakage_Rate'] = (monthly_leakage['Total_Leakage'] / monthly_leakage['Sales'] * 100)
        monthly_leakage.index = monthly_leakage.index.to_timestamp()

        if len(monthly_leakage) < 3:
            st.warning("Insufficient historical data for forecasting")
            return

        # Feature engineering for ML model
        monthly_leakage['Month'] = monthly_leakage.index.month
        monthly_leakage['Month_Num'] = range(len(monthly_leakage))

        # Train ML model
        features = ['Month', 'Month_Num']
        X = monthly_leakage[features]
        y = monthly_leakage['Total_Leakage']

        # Split data
        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]

        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Forecast future periods
        forecast_periods = st.slider("Forecast periods (months)", 1, 12, 6)

        last_month_num = monthly_leakage['Month_Num'].max()
        last_date = monthly_leakage.index.max()

        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                     periods=forecast_periods, freq='MS')

        future_features = pd.DataFrame({
            'Month': [d.month for d in future_dates],
            'Month_Num': range(last_month_num + 1, last_month_num + 1 + forecast_periods)
        })

        future_predictions = model.predict(future_features)

        # Visualize forecast
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(x=monthly_leakage.index, y=monthly_leakage['Total_Leakage'],
                               mode='lines+markers', name='Historical Leakage',
                               line=dict(color='#ff6b6b', width=2)))

        # Predictions
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions,
                               mode='lines+markers', name='Forecasted Leakage',
                               line=dict(color='#ffd93d', width=2, dash='dash')))

        fig.update_layout(title='Revenue Leakage Forecast',
                         xaxis_title='Date',
                         yaxis_title='Total Leakage ($)',
                         hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # Display forecast table
        st.markdown("#### Forecast Details")
        forecast_df = pd.DataFrame({
            'Period': future_dates.strftime('%Y-%m'),
            'Forecasted Leakage': future_predictions
        })

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(forecast_df.style.format({
                'Forecasted Leakage': '${:,.2f}'
            }))

        with col2:
            total_forecast_leakage = future_predictions.sum()
            avg_monthly_leakage = future_predictions.mean()

            st.metric("Total Forecasted Leakage", f"${total_forecast_leakage:,.2f}")
            st.metric("Avg Monthly Leakage", f"${avg_monthly_leakage:,.2f}")

            # Model performance
            if len(y_test) > 0:
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                st.metric("Model MAE", f"${mae:,.2f}")
                st.metric("Model RMSE", f"${rmse:,.2f}")

        # AI insights on forecast
        st.markdown("#### 🤖 AI-Generated Forecast Insights")
        forecast_prompt = f"""
        Analyze this revenue leakage forecast:

        Historical average monthly leakage: ${monthly_leakage['Total_Leakage'].mean():,.2f}
        Forecasted total leakage ({forecast_periods} months): ${total_forecast_leakage:,.2f}
        Forecasted average monthly leakage: ${avg_monthly_leakage:,.2f}

        Trend: {"increasing" if future_predictions[-1] > monthly_leakage['Total_Leakage'].mean() else "decreasing"}

        Provide:
        1. Analysis of the forecast trend
        2. Potential business impact
        3. Preventive action recommendations
        """

        with st.spinner("Generating forecast insights..."):
            insights = self.llm.conversational_response([{'sender': 'user', 'text': forecast_prompt}])['text']
        st.write(insights)

    def detect_anomalies(self):
        """Detect anomalous transactions that indicate revenue leakage"""
        st.subheader("📈 Anomaly Detection for Revenue Leakage")

        # Prepare features for anomaly detection
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

        # Select relevant features
        anomaly_features = []
        for col in ['Sales', 'Profit', 'Discount', 'Profit_Margin']:
            if col in numeric_cols:
                anomaly_features.append(col)

        if len(anomaly_features) < 2:
            st.warning("Insufficient numeric features for anomaly detection")
            return

        # Prepare data
        X = self.data[anomaly_features].dropna()

        # Train Isolation Forest
        contamination = st.slider("Anomaly Sensitivity (% of outliers)", 1, 20, 5) / 100

        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_predictions = iso_forest.fit_predict(X)

        # Add predictions to data
        anomalies_df = self.data.loc[X.index].copy()
        anomalies_df['Anomaly'] = anomaly_predictions
        anomalies_df['Anomaly_Label'] = anomalies_df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})

        # Count anomalies
        num_anomalies = (anomaly_predictions == -1).sum()
        anomaly_data = anomalies_df[anomalies_df['Anomaly'] == -1]

        st.metric("Anomalous Transactions Detected", num_anomalies)
        st.metric("Potential Revenue at Risk",
                 f"${anomaly_data['Sales'].sum():,.2f}" if len(anomaly_data) > 0 else "$0.00")

        # Visualize anomalies
        if 'Sales' in anomaly_features and 'Profit' in anomaly_features:
            fig = px.scatter(anomalies_df, x='Sales', y='Profit',
                           color='Anomaly_Label',
                           title='Anomaly Detection: Sales vs Profit',
                           color_discrete_map={'Normal': '#4ecdc4', 'Anomaly': '#ff6b6b'},
                           hover_data=['Discount'] if 'Discount' in anomalies_df.columns else None)
            st.plotly_chart(fig, use_container_width=True)

        # Show anomaly details
        if len(anomaly_data) > 0:
            st.markdown("#### Anomalous Transactions Details")

            display_cols = ['Sales', 'Profit', 'Discount', 'Profit_Margin']
            if 'Category' in anomaly_data.columns:
                display_cols.insert(0, 'Category')
            if 'Sub Category' in anomaly_data.columns:
                display_cols.insert(1, 'Sub Category')

            display_cols = [col for col in display_cols if col in anomaly_data.columns]

            st.dataframe(anomaly_data[display_cols].head(20).style.format({
                col: '${:,.2f}' for col in display_cols if col in ['Sales', 'Profit']
            }).format({
                col: '{:.2%}' for col in display_cols if col in ['Discount']
            }).format({
                col: '{:.2f}%' for col in display_cols if col in ['Profit_Margin']
            }))

            # Download anomalies
            csv = anomaly_data.to_csv(index=False)
            st.download_button(
                label="Download Anomalous Transactions",
                data=csv,
                file_name="revenue_leakage_anomalies.csv",
                mime="text/csv"
            )

    def generate_recommendations(self):
        """Generate AI-powered recommendations to reduce revenue leakage"""
        st.subheader("📋 AI-Powered Recommendations to Reduce Revenue Leakage")

        # Compile key findings
        total_sales = self.data['Sales'].sum() if 'Sales' in self.data.columns else 0
        total_profit = self.data['Profit'].sum() if 'Profit' in self.data.columns else 0
        avg_discount = self.data['Discount'].mean() * 100 if 'Discount' in self.data.columns else 0

        revenue_lost_discount = self.data['Revenue_Lost_to_Discount'].sum() if 'Revenue_Lost_to_Discount' in self.data.columns else 0
        negative_profit_trans = len(self.data[self.data['Profit'] < 0]) if 'Profit' in self.data.columns else 0

        # Category analysis
        category_insights = ""
        if 'Category' in self.data.columns:
            cat_profit = self.data.groupby('Category')['Profit'].sum().sort_values()
            worst_categories = cat_profit.head(3).to_string()
            best_categories = cat_profit.tail(3).to_string()
            category_insights = f"\nWorst performing categories:\n{worst_categories}\n\nBest performing categories:\n{best_categories}"

        # Generate comprehensive recommendations
        recommendation_prompt = f"""
        As a business consultant, provide detailed, actionable recommendations to reduce revenue leakage based on this analysis:

        **Financial Overview:**
        - Total Sales: ${total_sales:,.2f}
        - Total Profit: ${total_profit:,.2f}
        - Average Discount Rate: {avg_discount:.2f}%
        - Revenue Lost to Discounts: ${revenue_lost_discount:,.2f}
        - Number of Negative Profit Transactions: {negative_profit_trans}

        **Category Performance:**
        {category_insights}

        Please provide:

        1. **Immediate Actions** (0-30 days):
           - Quick wins to stop revenue leakage
           - Emergency measures for loss-making products/categories

        2. **Short-term Strategy** (1-3 months):
           - Pricing optimization recommendations
           - Discount policy improvements
           - Product mix optimization

        3. **Long-term Strategy** (3-12 months):
           - Strategic changes to business model
           - Category/product portfolio restructuring
           - Customer segment optimization

        4. **Specific Metrics to Track**:
           - KPIs to monitor improvement
           - Target values for each metric

        5. **Expected Impact**:
           - Estimated revenue recovery potential
           - Timeline for results

        Be specific, actionable, and quantify recommendations where possible.
        """

        with st.spinner("Generating comprehensive recommendations..."):
            recommendations = self.llm.conversational_response([{'sender': 'user', 'text': recommendation_prompt}])['text']

        st.write(recommendations)

        # Export report
        st.markdown("---")
        st.markdown("#### 📥 Export Complete Analysis Report")

        if st.button("Generate PDF Report"):
            st.info("PDF report generation feature - implementation requires additional PDF library")
            # This would require reportlab or similar library

        # Create summary report as CSV
        summary_data = {
            'Metric': [
                'Total Sales',
                'Total Profit',
                'Average Discount',
                'Revenue Lost to Discounts',
                'Negative Profit Transactions',
                'Total Revenue Leakage'
            ],
            'Value': [
                f"${total_sales:,.2f}",
                f"${total_profit:,.2f}",
                f"{avg_discount:.2f}%",
                f"${revenue_lost_discount:,.2f}",
                negative_profit_trans,
                f"${revenue_lost_discount + abs(self.data[self.data['Profit'] < 0]['Profit'].sum()):,.2f}"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        csv = summary_df.to_csv(index=False)

        st.download_button(
            label="Download Summary Report (CSV)",
            data=csv,
            file_name="revenue_leakage_summary.csv",
            mime="text/csv"
        )
