"""
Demand Forecasting ML Service
Handles 30/60/90-day forecasting for various metrics
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from datetime import timedelta
import pickle
import os
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class DemandForecaster:
    """ML-based demand forecasting service"""

    def __init__(self, model_path: str = "./app/ml/models"):
        self.model_path = model_path
        self.models = {}
        os.makedirs(model_path, exist_ok=True)

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for forecasting

        Args:
            data: DataFrame with date column and target values

        Returns:
            DataFrame with engineered features
        """
        df = data.copy()

        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            # Extract temporal features
            df['day_num'] = range(len(df))
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['quarter'] = df['date'].dt.quarter
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['week_of_year'] = df['date'].dt.isocalendar().week

            # Rolling averages
            if 'value' in df.columns:
                df['ma_7'] = df['value'].rolling(window=7, min_periods=1).mean()
                df['ma_30'] = df['value'].rolling(window=30, min_periods=1).mean()

        return df

    def train_model(
        self,
        data: pd.DataFrame,
        model_type: str = "gradient_boosting"
    ) -> Tuple[object, Dict[str, float]]:
        """
        Train forecasting model

        Args:
            data: DataFrame with features and target
            model_type: "gradient_boosting" or "random_forest"

        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        # Prepare features
        df = self.prepare_features(data)

        # Define features
        feature_cols = ['day_num', 'month', 'day_of_week', 'quarter',
                        'is_weekend', 'ma_7', 'ma_30']

        X = df[feature_cols].fillna(0)
        y = df['value']

        # Split data (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train model
        if model_type == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:  # random_forest
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )

        model.fit(X_train, y_train)

        # Calculate metrics
        y_pred = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mape": float(mean_absolute_percentage_error(y_test, y_pred) * 100),
            "r2": float(r2_score(y_test, y_pred))
        }

        logger.info(f"Model trained: {model_type}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")

        return model, metrics

    def generate_forecast(
        self,
        model: object,
        last_date: pd.Timestamp,
        last_day_num: int,
        periods: int,
        historical_ma: Dict[str, float]
    ) -> List[Dict]:
        """
        Generate forecast for specified periods

        Args:
            model: Trained ML model
            last_date: Last date in training data
            last_day_num: Last sequential day number
            periods: Number of days to forecast
            historical_ma: Last known moving averages

        Returns:
            List of forecast data points
        """
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods
        )

        forecast_df = pd.DataFrame({
            'date': future_dates,
            'day_num': range(last_day_num + 1, last_day_num + 1 + periods),
            'month': [d.month for d in future_dates],
            'day_of_week': [d.dayofweek for d in future_dates],
            'quarter': [d.quarter for d in future_dates],
            'is_weekend': [1 if d.dayofweek in [5, 6] else 0 for d in future_dates],
            'ma_7': historical_ma.get('ma_7', 0),
            'ma_30': historical_ma.get('ma_30', 0)
        })

        # Generate predictions
        feature_cols = ['day_num', 'month', 'day_of_week', 'quarter',
                        'is_weekend', 'ma_7', 'ma_30']

        forecast_df['value'] = model.predict(forecast_df[feature_cols])
        forecast_df['value'] = forecast_df['value'].clip(lower=0)  # Non-negative

        # Convert to list of dicts
        forecast_data = [
            {
                "date": row['date'].strftime('%Y-%m-%d'),
                "value": float(row['value'])
            }
            for _, row in forecast_df.iterrows()
        ]

        return forecast_data

    def forecast_demand(
        self,
        data: pd.DataFrame,
        forecast_type: str,
        periods: int = 90,
        model_type: str = "gradient_boosting"
    ) -> Dict:
        """
        Main forecasting method

        Args:
            data: DataFrame with historical data
            forecast_type: Type of forecast (service_volume, parts_demand, etc.)
            periods: Number of days to forecast
            model_type: ML model type

        Returns:
            Dictionary with forecast data and metrics
        """
        try:
            # Train model
            model, metrics = self.train_model(data, model_type)

            # Get last values for forecasting
            df = self.prepare_features(data)
            last_date = df['date'].max()
            last_day_num = df['day_num'].max()
            last_ma = {
                'ma_7': df['ma_7'].iloc[-1],
                'ma_30': df['ma_30'].iloc[-1]
            }

            # Generate forecast
            forecast_data = self.generate_forecast(
                model, last_date, last_day_num, periods, last_ma
            )

            # Save model
            model_filename = f"{forecast_type}_{model_type}.pkl"
            model_path = os.path.join(self.model_path, model_filename)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            return {
                "forecast_data": forecast_data,
                "model_metrics": metrics,
                "model_type": model_type,
                "forecast_type": forecast_type,
                "periods": periods
            }

        except Exception as e:
            logger.error(f"Forecasting error: {e}", exc_info=True)
            raise

    def load_models(self):
        """Load saved models from disk"""
        if not os.path.exists(self.model_path):
            logger.warning(f"Model path does not exist: {self.model_path}")
            return

        for filename in os.listdir(self.model_path):
            if filename.endswith('.pkl'):
                model_path = os.path.join(self.model_path, filename)
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                        model_name = filename.replace('.pkl', '')
                        self.models[model_name] = model
                        logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading model {filename}: {e}")
