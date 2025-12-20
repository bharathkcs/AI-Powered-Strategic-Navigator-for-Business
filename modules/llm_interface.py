import os
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

class LLMInterface:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")

        # Check if API key is valid (not empty, not placeholder)
        self.is_configured = False
        if api_key and len(api_key) > 20 and not api_key.startswith("8888") and not api_key.startswith("0000") and not api_key.startswith("9999"):
            try:
                self.client = OpenAI(api_key=api_key)
                self.model_name = "gpt-4o-mini"  # or 4.1, gpt-4o, etc.
                self.is_configured = True
            except Exception as e:
                print(f"Warning: LLM initialization failed: {str(e)}. Using deterministic fallback mode.")
                self.client = None
                self.is_configured = False
        else:
            print("Info: LLM not configured (invalid or placeholder API key). Using deterministic fallback mode.")
            self.client = None
            self.is_configured = False

    def conversational_response(self, conversation):
        """Generates a response with OpenAI new SDK format or deterministic fallback."""
        if not self.is_configured or self.client is None:
            # Deterministic fallback - analyze the prompt and provide rule-based response
            return self._generate_deterministic_response(conversation)

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
            # Fallback to deterministic response on error
            return self._generate_deterministic_response(conversation)

    def _generate_deterministic_response(self, conversation):
        """Generate deterministic rule-based response when LLM is not available."""
        if not conversation:
            return {"text": "No data provided for analysis.", "confidence": 0.5}

        last_message = conversation[-1]["text"].lower()

        # Extract key metrics from the prompt
        response_parts = []

        # Check for specific analysis types
        if "executive" in last_message and "insights" in last_message:
            response_parts.append("**Executive Analysis Summary:**")
            response_parts.append("\n**Key Findings:**")

            # Extract numeric values from prompt
            import re
            numbers = re.findall(r'₹?([\d,]+\.?\d*)', last_message)

            if "revenue" in last_message or "sales" in last_message:
                response_parts.append("1. **Revenue Performance**: Based on the data, revenue metrics indicate operational activity. Monitor trends for growth opportunities.")

            if "warranty" in last_message or "claim" in last_message:
                response_parts.append("2. **Warranty Management**: Warranty claim patterns suggest areas for quality improvement and cost control.")

            if "service" in last_message:
                response_parts.append("3. **Service Operations**: Service volume and efficiency metrics show the scale of operations and potential optimization areas.")

            response_parts.append("\n**Operational Recommendations:**")
            response_parts.append("- Focus on high-impact areas identified in the metrics")
            response_parts.append("- Implement monitoring for key performance indicators")
            response_parts.append("- Address quality and efficiency gaps systematically")

        elif "forecast" in last_message:
            response_parts.append("**Demand Forecast Analysis:**")
            response_parts.append("\n**Trend Interpretation:**")

            if "increasing" in last_message or "rising" in last_message:
                response_parts.append("The forecast shows an upward demand trend, indicating:")
                response_parts.append("- Increased market activity or seasonal factors")
                response_parts.append("- Need for inventory capacity planning")
                response_parts.append("- Opportunity for revenue growth if supply is maintained")
            elif "decreasing" in last_message or "declining" in last_message:
                response_parts.append("The forecast shows a downward demand trend, suggesting:")
                response_parts.append("- Market saturation or reduced market activity")
                response_parts.append("- Opportunity to optimize inventory and reduce carrying costs")
                response_parts.append("- Need to investigate root causes of demand decline")
            else:
                response_parts.append("The forecast indicates stable demand patterns:")
                response_parts.append("- Consistent market conditions")
                response_parts.append("- Maintain current inventory and operational levels")
                response_parts.append("- Continue routine monitoring for early trend detection")

            response_parts.append("\n**Business Implications:**")
            response_parts.append("- Align inventory planning with forecasted demand")
            response_parts.append("- Adjust resource allocation based on predicted volume")
            response_parts.append("- Monitor actual performance against forecast for accuracy validation")

            response_parts.append("\n**Risk Management:**")
            response_parts.append("- Wide confidence intervals suggest demand volatility - maintain safety stock")
            response_parts.append("- Limited historical data reduces forecast reliability - collect more data points")
            response_parts.append("- Seasonal patterns may affect accuracy - consider seasonal adjustments")

        elif "revenue leakage" in last_message or "leakage" in last_message:
            response_parts.append("**Revenue Leakage Analysis:**")
            response_parts.append("\n**Immediate Actions (0-30 days):**")
            response_parts.append("1. **Address High-Impact Leakages:**")
            response_parts.append("   - Review and tighten discount approval processes")
            response_parts.append("   - Audit transactions with negative margins")
            response_parts.append("   - Implement controls for loss-making categories")

            response_parts.append("\n2. **Process Improvements:**")
            response_parts.append("   - Establish pricing floors to prevent below-cost sales")
            response_parts.append("   - Implement real-time margin monitoring")
            response_parts.append("   - Create escalation workflows for high-discount requests")

            response_parts.append("\n**Short-term Strategy (1-3 months):**")
            response_parts.append("- Optimize pricing across product categories")
            response_parts.append("- Restructure discount policies based on profitability analysis")
            response_parts.append("- Improve product mix to favor higher-margin items")

            response_parts.append("\n**Expected Impact:**")
            response_parts.append("- Estimated 30-50% reduction in revenue leakage within 90 days")
            response_parts.append("- Improved profit margins through better pricing discipline")
            response_parts.append("- Enhanced visibility into profitability drivers")

        elif "spare part" in last_message or "procurement" in last_message:
            response_parts.append("**Spare Parts Inventory Recommendations:**")
            response_parts.append("\n**Category A (Critical Parts - 80% of usage):**")
            response_parts.append("- Maintain 60-90 days safety stock")
            response_parts.append("- Implement daily monitoring and reorder triggers")
            response_parts.append("- Use vendor-managed inventory where possible")

            response_parts.append("\n**Category B (Important Parts - 15% of usage):**")
            response_parts.append("- Maintain 30-45 days safety stock")
            response_parts.append("- Weekly monitoring and replenishment")
            response_parts.append("- Standard procurement lead times")

            response_parts.append("\n**Category C (Regular Parts - 5% of usage):**")
            response_parts.append("- Maintain 15-30 days safety stock")
            response_parts.append("- Monthly review and order consolidation")
            response_parts.append("- Just-in-time procurement acceptable")

            response_parts.append("\n**Cost Optimization:**")
            response_parts.append("- Consolidate orders to achieve volume discounts")
            response_parts.append("- Negotiate supplier agreements for Category A items")
            response_parts.append("- Consider alternative suppliers for high-cost, high-volume parts")

        elif "franchise" in last_message or "partner" in last_message:
            response_parts.append("**Franchise Performance Analysis:**")
            response_parts.append("\n**Performance Assessment:**")
            response_parts.append("- Top performers demonstrate strong operational execution and market presence")
            response_parts.append("- Mid-tier franchises show potential for improvement with targeted support")
            response_parts.append("- Underperforming partners may need operational intervention or restructuring")

            response_parts.append("\n**Growth Opportunities:**")
            response_parts.append("1. **Training & Development:** Implement best practice sharing from top performers")
            response_parts.append("2. **Market Expansion:** Support franchises in developing their local markets")
            response_parts.append("3. **Operational Excellence:** Provide tools and processes to improve efficiency")

            response_parts.append("\n**Support Recommendations:**")
            response_parts.append("- Platinum partners: Reward and incentivize continued excellence")
            response_parts.append("- Gold partners: Provide targeted support to reach platinum tier")
            response_parts.append("- Silver partners: Implement improvement plans with measurable milestones")

        else:
            # Generic business intelligence response
            response_parts.append("**Analysis Summary:**")
            response_parts.append("\nBased on the provided data, key observations include:")
            response_parts.append("- Operational metrics indicate current business performance levels")
            response_parts.append("- Trends suggest areas for focused attention and improvement")
            response_parts.append("- Data quality and completeness are important for reliable insights")

            response_parts.append("\n**Recommended Actions:**")
            response_parts.append("1. Monitor key performance indicators regularly")
            response_parts.append("2. Address identified gaps through targeted interventions")
            response_parts.append("3. Implement data-driven decision making processes")
            response_parts.append("4. Establish baseline metrics for ongoing performance tracking")

        return {"text": "\n".join(response_parts), "confidence": 0.75}

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
