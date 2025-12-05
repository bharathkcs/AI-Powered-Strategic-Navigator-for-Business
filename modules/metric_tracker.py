import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import seaborn as sns
from modules.llm_interface import LLMInterface

class MetricTracker:
    def __init__(self, llm, dataset):
        self.llm = llm
        self.dataset = dataset

    def automated_insight_generation(self, metric, data):
        """Generates insights using the LLM based on the plotted data."""
        prompt = (
            f"Analyze the following time series data for the metric '{metric}':\n\n"
            f"{data.to_string(index=False)}\n\n"
            "Provide a summary of key trends, patterns, and any anomalies detected."
        )
        response = self.llm.conversational_response([{'sender': 'user', 'text': prompt}])['text']
        return response

    def kpi_dashboard(self):
        """Displays an interactive dashboard for selected KPIs with threshold alerts."""
        st.subheader("Interactive KPI Dashboard")
        numeric_cols = self.dataset.select_dtypes(include='number').columns.tolist()
        selected_kpis = st.multiselect("Select KPIs to track", numeric_cols, default=numeric_cols[:3])

        thresholds = {}
        for kpi in selected_kpis:
            threshold = st.number_input(f"Set alert threshold for {kpi}", value=float(self.dataset[kpi].mean()))
            thresholds[kpi] = threshold

        fig, ax = plt.subplots()
        for kpi in selected_kpis:
            ax.plot(self.dataset.index, self.dataset[kpi], label=kpi)
            ax.axhline(y=thresholds[kpi], color='r', linestyle='--', label=f'{kpi} Threshold')
        ax.set_title('KPI Dashboard')
        ax.legend()
        st.pyplot(fig)

        
        alerts = [f"**Alert:** {kpi} has crossed the threshold of {thresholds[kpi]}!"
                  for kpi in selected_kpis if self.dataset[kpi].iloc[-1] > thresholds[kpi]]

        if alerts:
            st.warning('\n'.join(alerts))

    def correlation_analysis(self):
        """Identifies and visualizes correlations between different metrics."""
        st.subheader("Correlation Analysis Between Metrics")

        numeric_data = self.dataset.select_dtypes(include=['number'])
        if numeric_data.shape[1] > 1:
            corr = numeric_data.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            prompt = (
                f"Analyze the following correlation matrix:\n\n{corr.to_string()}\n\n"
                "Identify the strongest positive and negative correlations between metrics and discuss their potential implications."
            )
            response = self.llm.conversational_response([{'sender': 'user', 'text': prompt}])['text']
            st.write(f"**Correlation Insights:**\n{response}")
        else:
            st.write("Not enough numeric columns for correlation analysis.")

    def impact_analysis(self):
        """Analyzes the impact of implemented strategies on metrics over time."""
        st.subheader("Impact Analysis of Implemented Strategies")
        if 'Strategy_Implemented' in self.dataset.columns:
            strategy_dates = self.dataset[self.dataset['Strategy_Implemented'] == 1].index
            numeric_cols = self.dataset.select_dtypes(include='number').columns.tolist()
            fig, ax = plt.subplots()
            for col in numeric_cols:
                ax.plot(self.dataset.index, self.dataset[col], label=col)
            for date in strategy_dates:
                ax.axvline(x=date, color='green', linestyle='--', label='Strategy Implemented')
            ax.set_title('Impact of Strategies on Metrics')
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("No strategy implementation data found in the dataset.")

    def forecast_metrics(self):
        """Uses historical data to forecast future performance of key metrics."""
        st.subheader("Predictive Analytics for Key Metrics")
        numeric_cols = self.dataset.select_dtypes(include='number').columns.tolist()
        selected_metric = st.selectbox("Select a metric to forecast", numeric_cols)
        periods_to_forecast = st.slider("Select number of periods to forecast", 1, 12, 6)

        if st.button("Forecast Metric"):
            data = self.dataset[selected_metric].dropna().reset_index(drop=True)
            X = np.array(range(len(data))).reshape(-1, 1)
            y = data.values

            model = LinearRegression()
            model.fit(X, y)
            X_future = np.array(range(len(data), len(data) + periods_to_forecast)).reshape(-1, 1)
            y_future = model.predict(X_future)

            fig, ax = plt.subplots()
            ax.plot(data.index, y, label='Historical Data')
            ax.plot(X_future, y_future, label='Forecast', linestyle='--')
            ax.set_title(f"Forecast of {selected_metric}")
            ax.legend()
            st.pyplot(fig)

            forecast_data = pd.DataFrame({
                'Period': list(data.index) + list(X_future.flatten()),
                selected_metric: list(y) + list(y_future)
            })
            insight = self.automated_insight_generation(selected_metric, forecast_data[selected_metric])
            st.write(f"**Forecast Insights:**\n{insight}")

    def detect_anomalies(self):
        """Detects anomalies in key metrics using Isolation Forest."""
        st.subheader("Anomaly Detection in Metrics")
        numeric_cols = self.dataset.select_dtypes(include='number').columns.tolist()
        selected_metric = st.selectbox("Select a metric for anomaly detection", numeric_cols, key='anomaly_metric')

        if st.button("Detect Anomalies"):
            data = self.dataset[selected_metric].dropna().reset_index(drop=True)
            X = data.values.reshape(-1, 1)

            model = IsolationForest(contamination=0.1, random_state=42)
            preds = model.fit_predict(X)
            anomalies = data[preds == -1]

            fig, ax = plt.subplots()
            ax.plot(data.index, data.values, label='Data')
            ax.scatter(anomalies.index, anomalies.values, color='red', label='Anomalies')
            ax.set_title(f"Anomaly Detection in {selected_metric}")
            ax.legend()
            st.pyplot(fig)

            insight = self.automated_insight_generation(selected_metric, data)
            st.write(f"**Anomaly Detection Insights:**\n{insight}")

    def track_metrics(self):
        """Combines all metric tracking features."""
        self.kpi_dashboard()
        self.correlation_analysis()
        self.impact_analysis()
        self.forecast_metrics()
        self.detect_anomalies()
