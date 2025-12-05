"""AI-powered insights generation using OpenAI"""

import openai
from app.config import settings
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

openai.api_key = settings.OPENAI_API_KEY


async def generate_forecast_insights(
    forecast_type: str,
    forecast_data: List[Dict],
    metrics: Dict,
    period: int
) -> str:
    """Generate AI insights for forecasts"""

    if not settings.OPENAI_API_KEY:
        return "AI insights unavailable: OpenAI API key not configured"

    try:
        avg_forecast = sum([d['value'] for d in forecast_data]) / len(forecast_data)

        prompt = f"""
        Analyze this {period}-day {forecast_type} forecast for IFB's service ecosystem:

        Average forecasted value: {avg_forecast:.2f}
        Model accuracy (MAPE): {metrics.get('mape', 0):.2f}%
        Model R² score: {metrics.get('r2', 0):.3f}

        Provide:
        1. Trend interpretation (3-4 sentences)
        2. Business implications for IFB operations
        3. Top 3 actionable recommendations

        Keep it concise and actionable.
        """

        response = openai.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"AI insights generation error: {e}")
        return f"AI insights unavailable: {str(e)}"
