# modules/business_data_handler.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import plotly.express as px

class DataAnalyzer:
    def __init__(self, data, llm):
        self.data = data
        self.llm = llm

    def generate_insights(self):
        """Analyzes the dataset and generates automatic insights."""
        insights = []
        insights.append(f"Number of rows: {self.data.shape[0]}")
        insights.append(f"Number of columns: {self.data.shape[1]}")
        insights.append("\n### Column Information:")
        for column in self.data.columns:
            col_type = self.data[column].dtype
            col_unique = self.data[column].nunique()
            insights.append(f"- **{column}** (Type: {col_type}, Unique values: {col_unique})")

        numeric_cols = self.data.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            insights.append("\n### Numeric Column Statistics:")
            stats_summary = self.data[numeric_cols].describe().T
            insights.append(stats_summary.to_string())

        return "\n".join(insights)

    def automated_trend_analysis(self):
        """Uses LLM to analyze trends in the data."""
        numeric_cols = self.data.select_dtypes(include='number').columns.tolist()
        data_summary = self.data[numeric_cols].describe().transpose().round(2).to_string()
        prompt = f"Based on the following data summary, provide insights on any noticeable trends or patterns:\n{data_summary}"
        response = self.llm.conversational_response([{'sender': 'user', 'text': prompt}])['text']
        return response

    def generate_interactive_visualization(self, x_axis, y_axis, chart_type):
        """Generates interactive visualizations based on user selections."""
        if chart_type == "Line Chart":
            fig = px.line(self.data, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
        elif chart_type == "Bar Chart":
            fig = px.bar(self.data, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
        elif chart_type == "Scatter Plot":
            fig = px.scatter(self.data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        else:
            fig = None
        return fig

    def segment_analysis(self, segment_column):
        """Analyzes data by segments and provides insights."""
        if segment_column in self.data.columns:
            
            numeric_cols = self.data.select_dtypes(include='number').columns
            if numeric_cols.empty:
                return "No numeric columns available for analysis."

            segment_groups = self.data.groupby(segment_column)[numeric_cols].mean().head(5)
            st.write(segment_groups)

            segment_summary = segment_groups.round(2).to_string()
            max_length = 1000  
            if len(segment_summary) > max_length:
                segment_summary = segment_summary[:max_length] + "\n... [Data truncated]"

            prompt = (
                f"Analyze the differences between the following segments (showing top 5 segments):\n"
                f"{segment_summary}\n\n"
                "Provide insights on the differences between these segments."
            )
            response = self.llm.conversational_response([{'sender': 'user', 'text': prompt}])['text']
            return response
        else:
            return "Selected segment column is not in the dataset."



    def time_series_analysis(self):
        """Performs time series analysis if date columns are present."""
        date_columns = self.data.select_dtypes(include=['datetime', 'object']).columns
        for col in date_columns:
            try:
                self.data[col] = pd.to_datetime(self.data[col])
                time_cols = [col for col in date_columns if pd.api.types.is_datetime64_any_dtype(self.data[col])]
                if time_cols:
                    for time_col in time_cols:
                        numeric_cols = self.data.select_dtypes(include='number').columns
                        for num_col in numeric_cols:
                            fig = px.line(self.data, x=time_col, y=num_col, title=f"{num_col} over {time_col}")
                            st.plotly_chart(fig)
                            
                            sample_data = self.data[[time_col, num_col]].dropna().head(100)
                            data_string = sample_data.to_string(index=False)
                            max_length = 1000
                            if len(data_string) > max_length:
                                data_string = data_string[:max_length] + "\n... [Data truncated]"
                            prompt = f"Analyze the trend of {num_col} over time based on the following data:\n{data_string}"
                            response = self.llm.conversational_response([{'sender': 'user', 'text': prompt}])['text']
                            st.write(response)
                break  
            except Exception:
                continue


    def key_findings_summary(self):
        """Generates a summary of key findings in the data."""
        data_summary = self.data.describe(include='all').transpose().round(2).to_string()
        max_length = 1500  
        if len(data_summary) > max_length:
            data_summary = data_summary[:max_length] + "\n... [Data truncated]"
        prompt = f"Based on the following data summary, provide a concise summary of key findings:\n{data_summary}"
        response = self.llm.conversational_response([{'sender': 'user', 'text': prompt}])['text']
        st.write(response)

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
