import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.llm_interface import LLMInterface

class StrategyMap:
    def __init__(self, dataset, llm):
        self.dataset = dataset
        self.llm = llm

    def generate_scenarios(self, strategy_input):
        """Generates different strategic scenarios and their potential impact."""
        prompt = (
            f"Given the following dataset summary:\n{self.dataset.describe(include='all').to_string()}\n\n"
            f"Analyze the potential impact of the following strategy: {strategy_input}\n"
            f"Provide a detailed projection of key business metrics such as Sales, Revenue, Profit, Customer Satisfaction, and Market Share over the next 12 months."
        )
        response = self.llm.conversational_response([{'sender': 'user', 'text': prompt}])['text']
        return response

    def display_interactive_scenario(self):
        """Allows users to input strategy parameters and visualizes projected outcomes."""
        st.subheader("Interactive Scenario Planning")
        strategy_input = st.text_area("Describe your proposed strategy:", "Increase marketing spend by 20% targeting young adults.")
        if st.button("Generate Scenario"):
            with st.spinner("Generating scenario..."):
                scenario_analysis = self.generate_scenarios(strategy_input)
            st.write(f"**Scenario Analysis:**\n{scenario_analysis}")

            
            months = pd.date_range(start=pd.Timestamp.today(), periods=12, freq='M')
            metrics = ['Sales', 'Revenue', 'Profit', 'Customer Satisfaction', 'Market Share']
            data = pd.DataFrame(index=months, columns=metrics)

           
            np.random.seed(0)
            for metric in metrics:
                base_value = np.random.uniform(100, 200)
                growth_rate = np.random.uniform(0.02, 0.05)
                data[metric] = [base_value * (1 + growth_rate) ** i for i in range(12)]

            fig = go.Figure()
            for metric in metrics:
                fig.add_trace(go.Scatter(x=data.index, y=data[metric], mode='lines+markers', name=metric))
            fig.update_layout(title='Projected Metrics over Next 12 Months', xaxis_title='Month', yaxis_title='Value')
            st.plotly_chart(fig)
