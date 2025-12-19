"""
Spare Parts Demand Forecasting and Service-Led Revenue Leakage Analysis

This module provides enterprise Decision Intelligence for IFB Industries focusing on:
- Spare parts planning
- Demand forecasting
- Service-led revenue leakage detection

CRITICAL DESIGN PRINCIPLES:
1. Schema-driven: No hardcoded column names
2. Semantic inference: Pattern-based column matching
3. Fail-fast: Explicit errors on schema mismatch
4. Deterministic: Reproducible outputs
5. Future-proof: Works with changing column names
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determinism
GLOBAL_RANDOM_SEED = 42
np.random.seed(GLOBAL_RANDOM_SEED)


class SchemaInferenceError(Exception):
    """Raised when required canonical fields cannot be inferred from schema"""
    pass


def resolve_column_after_merge(df: pd.DataFrame, base_col: Optional[str]) -> str:
    """
    Resolve column name after pandas merge when suffixes are applied.

    Example:
        ZZBRNCH → ZZBRNCH_consumed

    Rules:
    - Exact match wins
    - Else: first column that startswith(base_col) wins
    - Else: fail fast
    """
    if base_col is None or str(base_col).strip() == "":
        raise SchemaInferenceError(
            "Base column name is None/empty while resolving post-merge columns. "
            "This indicates schema inference did not produce a required field."
        )

    if base_col in df.columns:
        return base_col

    candidates = [c for c in df.columns if str(c).startswith(str(base_col))]
    if candidates:
        return candidates[0]

    raise SchemaInferenceError(
        f"Column '{base_col}' not found after merge. "
        f"Available columns: {list(df.columns)}"
    )


class SparePartsForecastingEngine:
    """
    Enterprise spare parts demand forecasting and revenue leakage detection engine.

    This class implements a schema-driven approach that:
    - Infers column semantics automatically
    - Performs data quality validation
    - Generates demand forecasts (30/60/90 day)
    - Detects service-led revenue leakage
    """

    def __init__(self, excel_file_path: str = None, excel_data: Dict[str, pd.DataFrame] = None):
        """
        Initialize the forecasting engine.

        Args:
            excel_file_path: Path to Excel file with 4 sheets
            excel_data: Pre-loaded dictionary of DataFrames {sheet_name: df}
        """
        self.excel_path = excel_file_path
        self.excel_data = excel_data

        # Canonical field mappings
        self.canonical_fields: Dict[str, str] = {}

        # Data containers
        self.indent_df: Optional[pd.DataFrame] = None
        self.consumed_df: Optional[pd.DataFrame] = None
        self.branches_df: Optional[pd.DataFrame] = None
        self.franchises_df: Optional[pd.DataFrame] = None

        # Processed data
        self.normalized_clean_data: Optional[pd.DataFrame] = None
        self.forecast_30_60_90: Optional[pd.DataFrame] = None
        self.branch_leakage_summary: Optional[pd.DataFrame] = None
        self.franchise_leakage_summary: Optional[pd.DataFrame] = None
        self.top_20_high_risk_spares: Optional[pd.DataFrame] = None

        # Metadata
        self.join_loss_percentage: float = 0.0
        self.data_quality_stats: Dict[str, Any] = {}

    # ----------------------------
    # Loading
    # ----------------------------
    def load_data(self) -> None:
        """Load Excel sheets into DataFrames"""
        if self.excel_data:
            self.indent_df = self.excel_data.get("INDENT")
            self.consumed_df = self.excel_data.get("SPARES_CONSUMED")
            self.branches_df = self.excel_data.get("BRANCHES")
            self.franchises_df = self.excel_data.get("FRANCHISES")
        elif self.excel_path:
            logger.info(f"Loading data from {self.excel_path}")
            excel_file = pd.ExcelFile(self.excel_path)

            self.indent_df = pd.read_excel(excel_file, sheet_name="INDENT")
            self.consumed_df = pd.read_excel(excel_file, sheet_name="SPARES_CONSUMED")
            self.branches_df = pd.read_excel(excel_file, sheet_name="BRANCHES")
            self.franchises_df = pd.read_excel(excel_file, sheet_name="FRANCHISES")
        else:
            raise ValueError("Either excel_file_path or excel_data must be provided")

        # Fail-fast if any required sheet missing
        if self.indent_df is None or self.consumed_df is None:
            raise SchemaInferenceError(
                "Missing required sheets. Ensure INDENT and SPARES_CONSUMED exist."
            )
        if self.branches_df is None or self.franchises_df is None:
            logger.warning(
                "BRANCHES or FRANCHISES sheet missing. Enrichment will be skipped for missing references."
            )

        logger.info(f"Loaded INDENT: {len(self.indent_df)} rows")
        logger.info(f"Loaded SPARES_CONSUMED: {len(self.consumed_df)} rows")
        logger.info(f"Loaded BRANCHES: {len(self.branches_df) if self.branches_df is not None else 0} rows")
        logger.info(f"Loaded FRANCHISES: {len(self.franchises_df) if self.franchises_df is not None else 0} rows")

    # ----------------------------
    # Schema inference
    # ----------------------------
    def infer_canonical_fields(self) -> Dict[str, str]:
        """
        Infer canonical field names from actual column names using pattern matching.

        Returns:
            Dictionary mapping canonical names to actual column names

        Raises:
            SchemaInferenceError: If required fields cannot be inferred
        """
        logger.info("Starting semantic column inference...")

        inference_rules = {
            # INDENT sheet fields
            "job_id": ["OBJECT"],
            "posting_date": ["POSTING"],
            "branch_code": ["BRNCH"],
            "franchise_code": ["FRNCH", "PARTNER"],
            "ordered_part_id": ["ORDERED_PROD"],
            "ordered_qty": ["QTY"],
            "item_description": ["ITEM_DESCRIPTION", "DESCRIPTION"],
            "eta_date": ["ETA"],

            # SPARES_CONSUMED sheet fields
            "consumed_part_id": ["PRODUCT_ID"],
            "process_type": ["PROCESS"],
            "machine_status": ["MACHINE"],
            "closing_date": ["CLOSING"],
            "material_group": ["MAT_GRP"],

            # Reference sheet fields
            "branch_name": ["Description"],
            "franchise_name": ["MC_NAME"],
        }

        canonical_mapping: Dict[str, str] = {}

        # Infer INDENT fields
        if self.indent_df is not None:
            for canonical, patterns in inference_rules.items():
                if canonical in [
                    "job_id",
                    "posting_date",
                    "branch_code",
                    "franchise_code",
                    "ordered_part_id",
                    "ordered_qty",
                    "item_description",
                    "eta_date",
                ]:
                    actual_col = self._find_column(self.indent_df.columns, patterns)
                    if actual_col:
                        canonical_mapping[f"indent_{canonical}"] = actual_col

        # Infer SPARES_CONSUMED fields
        if self.consumed_df is not None:
            for canonical, patterns in inference_rules.items():
                actual_col = self._find_column(self.consumed_df.columns, patterns)
                if actual_col:
                    canonical_mapping[f"consumed_{canonical}"] = actual_col

        # Infer BRANCHES fields
        if self.branches_df is not None:
            branch_code_col = self._find_column(self.branches_df.columns, ["BRANCH", "CODE"])
            if branch_code_col:
                canonical_mapping["branch_code_ref"] = branch_code_col

            branch_name_col = self._find_column(self.branches_df.columns, ["Description", "NAME"])
            if branch_name_col:
                canonical_mapping["branch_name"] = branch_name_col

        # Infer FRANCHISES fields
        if self.franchises_df is not None:
            franchise_code_col = self._find_column(self.franchises_df.columns, ["PARTNER", "FRANCHISE"])
            if franchise_code_col:
                canonical_mapping["franchise_code_ref"] = franchise_code_col

            franchise_name_col = self._find_column(self.franchises_df.columns, ["MC_NAME", "NAME"])
            if franchise_name_col:
                canonical_mapping["franchise_name"] = franchise_name_col

        # Validate required fields (kept exactly as your pipeline expects)
        required_fields = [
            "indent_job_id",
            "indent_posting_date",
            "indent_branch_code",
            "indent_franchise_code",
            "indent_ordered_part_id",
            "consumed_job_id",
            "consumed_posting_date",
            "consumed_consumed_part_id",
        ]

        missing_fields = [f for f in required_fields if f not in canonical_mapping]

        if missing_fields:
            error_msg = f"SCHEMA INFERENCE FAILED. Cannot infer required fields: {missing_fields}\n"
            error_msg += f"Available INDENT columns: {list(self.indent_df.columns)}\n"
            error_msg += f"Available CONSUMED columns: {list(self.consumed_df.columns)}"
            raise SchemaInferenceError(error_msg)

        self.canonical_fields = canonical_mapping
        logger.info(f"Successfully inferred {len(canonical_mapping)} canonical fields")
        return canonical_mapping

    def _find_column(self, columns: pd.Index, patterns: List[str]) -> Optional[str]:
        """Find column matching any of the patterns (case-insensitive, partial match)"""
        for col in columns:
            col_upper = str(col).upper()
            for pattern in patterns:
                if pattern.upper() in col_upper:
                    return col
        return None

    # ----------------------------
    # Cleaning
    # ----------------------------
    def clean_and_validate_data(self) -> None:
        """
        Clean and validate data with quality flagging.

        Adds 'data_quality_flag' column with values:
        - 'clean': Valid record
        - 'missing_quantity': Missing quantity in AMC process
        - 'date_error': Date parsing failed
        - 'duplicate': Duplicate record
        - 'invalid': Invalid quantity (negative)
        """
        logger.info("Starting data cleaning and validation...")

        self.indent_df = self._clean_indent_data()
        self.consumed_df = self._clean_consumed_data()

        logger.info("Data cleaning completed")

    def _clean_indent_data(self) -> pd.DataFrame:
        """Clean and validate INDENT data"""
        df = self.indent_df.copy()
        df["data_quality_flag"] = "clean"

        date_col = self.canonical_fields.get("indent_posting_date")
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df.loc[df[date_col].isna(), "data_quality_flag"] = "date_error"

        qty_col = self.canonical_fields.get("indent_ordered_qty")
        if qty_col and qty_col in df.columns:
            df.loc[pd.to_numeric(df[qty_col], errors="coerce") < 0, "data_quality_flag"] = "invalid"

        # Remove duplicates
        dedup_cols = [
            self.canonical_fields.get("indent_job_id"),
            self.canonical_fields.get("indent_ordered_part_id"),
            self.canonical_fields.get("indent_posting_date"),
            self.canonical_fields.get("indent_branch_code"),
        ]
        dedup_cols = [c for c in dedup_cols if c is not None and c in df.columns]

        if dedup_cols:
            duplicates = df.duplicated(subset=dedup_cols, keep="first")
            df.loc[duplicates, "data_quality_flag"] = "duplicate"

        quality_counts = df["data_quality_flag"].value_counts()
        logger.info(f"INDENT quality distribution: {quality_counts.to_dict()}")

        return df

    def _clean_consumed_data(self) -> pd.DataFrame:
        """Clean and validate SPARES_CONSUMED data"""
        df = self.consumed_df.copy()
        df["data_quality_flag"] = "clean"

        date_col = self.canonical_fields.get("consumed_posting_date")
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df.loc[df[date_col].isna(), "data_quality_flag"] = "date_error"

        closing_col = self.canonical_fields.get("consumed_closing_date")
        if closing_col and closing_col in df.columns:
            df[closing_col] = pd.to_datetime(df[closing_col], errors="coerce")

        # Check for missing quantity in AMC
        process_col = self.canonical_fields.get("consumed_process_type")
        part_col = self.canonical_fields.get("consumed_consumed_part_id")

        if process_col and part_col and process_col in df.columns and part_col in df.columns:
            proc = df[process_col].astype(str).str.strip().str.upper()
            amc_missing = (proc == "AMC") & (df[part_col].isna())
            df.loc[amc_missing, "data_quality_flag"] = "missing_quantity"

        # Remove duplicates
        dedup_cols = [
            self.canonical_fields.get("consumed_job_id"),
            self.canonical_fields.get("consumed_consumed_part_id"),
            self.canonical_fields.get("consumed_posting_date"),
        ]
        dedup_cols = [c for c in dedup_cols if c is not None and c in df.columns]

        if dedup_cols:
            duplicates = df.duplicated(subset=dedup_cols, keep="first")
            df.loc[duplicates, "data_quality_flag"] = "duplicate"

        quality_counts = df["data_quality_flag"].value_counts()
        logger.info(f"CONSUMED quality distribution: {quality_counts.to_dict()}")

        return df

    # ----------------------------
    # Integration
    # ----------------------------
    def integrate_data(self) -> pd.DataFrame:
        """
        Integrate INDENT and SPARES_CONSUMED data with reference sheets.

        CRITICAL:
        - After merge, pandas suffixes duplicate columns like data_quality_flag -> data_quality_flag_consumed/indent
        - We MUST restore a canonical data_quality_flag for downstream steps
        """
        logger.info("Starting data integration...")

        indent_job_col = self.canonical_fields.get("indent_job_id")
        consumed_job_col = self.canonical_fields.get("consumed_job_id")

        indent_clean = self.indent_df[self.indent_df["data_quality_flag"] != "duplicate"].copy()
        consumed_clean = self.consumed_df[self.consumed_df["data_quality_flag"] != "duplicate"].copy()

        if indent_job_col and consumed_job_col:
            merged = pd.merge(
                consumed_clean,
                indent_clean,
                left_on=consumed_job_col,
                right_on=indent_job_col,
                how="left",
                suffixes=("_consumed", "_indent"),
            )

            total_consumed = len(consumed_clean)
            matched = merged[indent_job_col].notna().sum() if indent_job_col in merged.columns else 0
            self.join_loss_percentage = (
                ((total_consumed - matched) / total_consumed * 100) if total_consumed > 0 else 0
            )
            logger.info(f"Join loss percentage: {self.join_loss_percentage:.2f}%")
        else:
            merged = consumed_clean
            self.join_loss_percentage = 100.0

        # ✅ FIX: Restore canonical data_quality_flag after merge suffixing
        if "data_quality_flag" not in merged.columns:
            if "data_quality_flag_consumed" in merged.columns:
                merged["data_quality_flag"] = merged["data_quality_flag_consumed"]
            elif "data_quality_flag_indent" in merged.columns:
                merged["data_quality_flag"] = merged["data_quality_flag_indent"]
            else:
                merged["data_quality_flag"] = "clean"

        # Attach branch names (robust guards)
        if self.branches_df is not None:
            consumed_branch_base = self.canonical_fields.get("consumed_branch_code")
            if consumed_branch_base:
                branch_code_col = resolve_column_after_merge(merged, consumed_branch_base)
                branch_ref_col = self.canonical_fields.get("branch_code_ref")
                branch_name_col = self.canonical_fields.get("branch_name")

                if (
                    branch_code_col
                    and branch_ref_col
                    and branch_name_col
                    and branch_ref_col in self.branches_df.columns
                    and branch_name_col in self.branches_df.columns
                ):
                    merged = pd.merge(
                        merged,
                        self.branches_df[[branch_ref_col, branch_name_col]],
                        left_on=branch_code_col,
                        right_on=branch_ref_col,
                        how="left",
                    )

        # Attach franchise names (robust guards)
        if self.franchises_df is not None:
            consumed_franchise_base = self.canonical_fields.get("consumed_franchise_code")
            if consumed_franchise_base:
                franchise_code_col = resolve_column_after_merge(merged, consumed_franchise_base)
                franchise_ref_col = self.canonical_fields.get("franchise_code_ref")
                franchise_name_col = self.canonical_fields.get("franchise_name")

                if (
                    franchise_code_col
                    and franchise_ref_col
                    and franchise_name_col
                    and franchise_ref_col in self.franchises_df.columns
                    and franchise_name_col in self.franchises_df.columns
                ):
                    merged = pd.merge(
                        merged,
                        self.franchises_df[[franchise_ref_col, franchise_name_col]],
                        left_on=franchise_code_col,
                        right_on=franchise_ref_col,
                        how="left",
                    )

        # ✅ Extra safety: ensure data_quality_flag survived any later merges too
        if "data_quality_flag" not in merged.columns:
            if "data_quality_flag_consumed" in merged.columns:
                merged["data_quality_flag"] = merged["data_quality_flag_consumed"]
            else:
                merged["data_quality_flag"] = "clean"

        logger.info(f"Integration complete. Final dataset: {len(merged)} rows")

        self.normalized_clean_data = merged
        return merged

    # ----------------------------
    # Feature engineering
    # ----------------------------
    def engineer_features(self) -> pd.DataFrame:
        """
        Engineer features for spare parts analysis.

        Features created:
        - ordered_qty, consumed_qty
        - demand_gap
        - order_fulfillment_ratio
        - lead_time_days
        - monthly_consumption
        - rolling_3M_consumption, rolling_6M_consumption
        - demand_volatility
        - repeat_consumption_flag
        """
        logger.info("Starting feature engineering...")

        if self.normalized_clean_data is None:
            raise ValueError("normalized_clean_data is None. Run integrate_data() before engineer_features().")

        df = self.normalized_clean_data.copy()

        posting_date_col = resolve_column_after_merge(df, self.canonical_fields.get("consumed_posting_date"))

        closing_date_col = (
            resolve_column_after_merge(df, self.canonical_fields.get("consumed_closing_date"))
            if self.canonical_fields.get("consumed_closing_date")
            else None
        )

        ordered_qty_col = (
            resolve_column_after_merge(df, self.canonical_fields.get("indent_ordered_qty"))
            if self.canonical_fields.get("indent_ordered_qty")
            else None
        )

        consumed_part_col = resolve_column_after_merge(df, self.canonical_fields.get("consumed_consumed_part_id"))

        # Basic quantity features
        if ordered_qty_col and ordered_qty_col in df.columns:
            df["ordered_qty"] = pd.to_numeric(df[ordered_qty_col], errors="coerce").fillna(0)
        else:
            df["ordered_qty"] = 0

        df["consumed_qty"] = 1  # each row represents one consumption event
        df["demand_gap"] = df["ordered_qty"] - df["consumed_qty"]

        df["order_fulfillment_ratio"] = np.where(
            df["ordered_qty"] > 0,
            df["consumed_qty"] / df["ordered_qty"],
            np.nan,
        )

        # Lead time
        if posting_date_col and closing_date_col and posting_date_col in df.columns and closing_date_col in df.columns:
            df["lead_time_days"] = (
                pd.to_datetime(df[closing_date_col], errors="coerce")
                - pd.to_datetime(df[posting_date_col], errors="coerce")
            ).dt.days
        else:
            df["lead_time_days"] = np.nan

        # Time-based features
        if posting_date_col and posting_date_col in df.columns:
            df[posting_date_col] = pd.to_datetime(df[posting_date_col], errors="coerce")
            df["year"] = df[posting_date_col].dt.year
            df["month"] = df[posting_date_col].dt.month
            df["quarter"] = df[posting_date_col].dt.quarter
            df["year_month"] = df[posting_date_col].dt.to_period("M")

        # Monthly consumption aggregation
        if consumed_part_col and posting_date_col and "year_month" in df.columns:
            monthly_agg = (
                df.groupby([consumed_part_col, "year_month"])
                .agg({"consumed_qty": "sum"})
                .reset_index()
            )
            monthly_agg.columns = [consumed_part_col, "year_month", "monthly_consumption"]

            df = pd.merge(df, monthly_agg, on=[consumed_part_col, "year_month"], how="left")

        # Rolling consumption and volatility (simplified)
        if consumed_part_col and consumed_part_col in df.columns:
            part_consumption = (
                df.groupby(consumed_part_col)
                .agg({"consumed_qty": ["sum", "mean", "std", "count"]})
                .reset_index()
            )
            part_consumption.columns = [
                consumed_part_col,
                "total_consumption",
                "avg_consumption",
                "std_consumption",
                "consumption_count",
            ]

            part_consumption["demand_volatility"] = np.where(
                part_consumption["avg_consumption"] > 0,
                (part_consumption["std_consumption"].fillna(0) / part_consumption["avg_consumption"]),
                0,
            )

            part_consumption["repeat_consumption_flag"] = part_consumption["consumption_count"] > 1
            df = pd.merge(df, part_consumption, on=consumed_part_col, how="left")

        df["rolling_3M_consumption"] = df.get("monthly_consumption", 0).fillna(0) * 3
        df["rolling_6M_consumption"] = df.get("monthly_consumption", 0).fillna(0) * 6

        # Safety: ensure flag exists for downstream even if something weird happens
        if "data_quality_flag" not in df.columns:
            df["data_quality_flag"] = "clean"

        logger.info("Feature engineering completed")
        self.normalized_clean_data = df
        return df

    # ----------------------------
    # Forecasting
    # ----------------------------
    def generate_demand_forecast(self) -> pd.DataFrame:
        """
        Generate 30/60/90-day demand forecasts for spare parts.

        Granularity: part, branch, franchise (if available)
        Method: Ensemble of RandomForest, GradientBoosting, LinearRegression

        Returns:
            DataFrame with forecasts and confidence intervals
        """
        logger.info("Starting demand forecasting...")

        if self.normalized_clean_data is None:
            raise ValueError("normalized_clean_data is None. Run engineer_features() before generate_demand_forecast().")

        df = self.normalized_clean_data.copy()

        # Must exist
        if "data_quality_flag" not in df.columns:
            raise SchemaInferenceError("Missing 'data_quality_flag'. Run clean_and_validate_data() first.")

        # Resolve actual column names
        part_col = resolve_column_after_merge(df, self.canonical_fields.get("consumed_consumed_part_id"))
        date_col = resolve_column_after_merge(df, self.canonical_fields.get("consumed_posting_date"))

        # Optional
        branch_col = None
        franchise_col = None
        if self.canonical_fields.get("consumed_branch_code"):
            branch_col = resolve_column_after_merge(df, self.canonical_fields.get("consumed_branch_code"))
        if self.canonical_fields.get("consumed_franchise_code"):
            franchise_col = resolve_column_after_merge(df, self.canonical_fields.get("consumed_franchise_code"))

        df_clean = df[df["data_quality_flag"] == "clean"].copy()
        if df_clean.empty:
            logger.warning("No clean records available for forecasting. Returning empty forecast.")
            self.forecast_30_60_90 = pd.DataFrame()
            return self.forecast_30_60_90

        df_clean["date"] = pd.to_datetime(df_clean[date_col], errors="coerce")
        df_clean = df_clean[df_clean["date"].notna()]
        if df_clean.empty:
            logger.warning("All clean records have invalid dates. Returning empty forecast.")
            self.forecast_30_60_90 = pd.DataFrame()
            return self.forecast_30_60_90

        df_clean["year_month"] = df_clean["date"].dt.to_period("M")

        groupby_cols = [part_col, "year_month"]
        if branch_col and branch_col in df_clean.columns:
            groupby_cols.append(branch_col)
        if franchise_col and franchise_col in df_clean.columns:
            groupby_cols.append(franchise_col)

        monthly_demand = (
            df_clean.groupby(groupby_cols)
            .agg({"consumed_qty": "sum"})
            .reset_index()
        )
        monthly_demand.columns = groupby_cols + ["demand"]

        forecasts: List[Dict[str, Any]] = []

        for part in monthly_demand[part_col].dropna().unique():
            part_data = monthly_demand[monthly_demand[part_col] == part].copy()
            if len(part_data) < 3:
                continue

            part_data = part_data.sort_values("year_month")
            part_data["month_index"] = range(len(part_data))
            part_data["demand_lag1"] = part_data["demand"].shift(1)
            part_data["demand_lag2"] = part_data["demand"].shift(2)
            part_data["demand_ma3"] = part_data["demand"].rolling(window=3, min_periods=1).mean()

            part_data_clean = part_data.dropna()
            if len(part_data_clean) < 2:
                continue

            feature_cols = ["month_index", "demand_lag1", "demand_lag2", "demand_ma3"]
            X = part_data_clean[feature_cols].values
            y = part_data_clean["demand"].values

            try:
                rf_model = RandomForestRegressor(
                    n_estimators=50,
                    random_state=GLOBAL_RANDOM_SEED,
                    max_depth=5,
                )
                gb_model = GradientBoostingRegressor(
                    n_estimators=50,
                    random_state=GLOBAL_RANDOM_SEED,
                    max_depth=3,
                )
                lr_model = LinearRegression()

                rf_model.fit(X, y)
                gb_model.fit(X, y)
                lr_model.fit(X, y)

                last_month_index = int(part_data_clean["month_index"].max())
                last_demand = float(part_data_clean["demand"].iloc[-1])
                last_demand_lag1 = float(part_data_clean["demand_lag1"].iloc[-1])
                last_demand_ma3 = float(part_data_clean["demand_ma3"].iloc[-1])

                for i in range(1, 4):  # 1, 2, 3 months
                    X_future = np.array([[
                        last_month_index + i,
                        last_demand,
                        last_demand_lag1,
                        last_demand_ma3,
                    ]])

                    rf_pred = float(rf_model.predict(X_future)[0])
                    gb_pred = float(gb_model.predict(X_future)[0])
                    lr_pred = float(lr_model.predict(X_future)[0])

                    ensemble_pred = (rf_pred + gb_pred + lr_pred) / 3.0
                    std_pred = float(np.std([rf_pred, gb_pred, lr_pred]))
                    ci_lower = max(0.0, ensemble_pred - 1.96 * std_pred)
                    ci_upper = ensemble_pred + 1.96 * std_pred

                    branch_val = part_data[branch_col].iloc[0] if branch_col and branch_col in part_data.columns else "ALL"
                    franchise_val = part_data[franchise_col].iloc[0] if franchise_col and franchise_col in part_data.columns else "ALL"

                    hist_std = part_data["demand"].std()
                    forecasts.append({
                        "part_id": part,
                        "branch": branch_val,
                        "franchise": franchise_val,
                        "forecast_horizon": f"{i * 30}_day",
                        "forecast_demand": ensemble_pred,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "historical_avg_demand": float(part_data["demand"].mean()),
                        "historical_std_demand": float(hist_std) if hist_std is not None and not np.isnan(hist_std) else 0.0,
                    })

            except Exception as e:
                logger.warning(f"Forecast failed for part {part}: {str(e)}")
                continue

        forecast_df = pd.DataFrame(forecasts)
        logger.info(f"Generated forecasts for {len(forecast_df)} part-horizon combinations")

        self.forecast_30_60_90 = forecast_df
        return forecast_df

    # ----------------------------
    # Leakage detection
    # ----------------------------
    def detect_revenue_leakage(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Detect service-led revenue leakage.

        Leakage indicators:
        - Excess consumption (beyond normal patterns)
        - Repeat part replacement (failure rate)
        - Abnormal warranty behavior
        - Stock mismatch
        """
        logger.info("Starting revenue leakage detection...")

        if self.normalized_clean_data is None:
            raise ValueError("normalized_clean_data is None. Run engineer_features() before detect_revenue_leakage().")

        df = self.normalized_clean_data.copy()

        part_col = resolve_column_after_merge(df, self.canonical_fields.get("consumed_consumed_part_id"))
        job_col = resolve_column_after_merge(df, self.canonical_fields.get("consumed_job_id"))

        branch_col = None
        franchise_col = None
        process_col = None

        if self.canonical_fields.get("consumed_branch_code"):
            branch_col = resolve_column_after_merge(df, self.canonical_fields.get("consumed_branch_code"))
        if self.canonical_fields.get("consumed_franchise_code"):
            franchise_col = resolve_column_after_merge(df, self.canonical_fields.get("consumed_franchise_code"))
        if self.canonical_fields.get("consumed_process_type"):
            process_col = resolve_column_after_merge(df, self.canonical_fields.get("consumed_process_type"))

        # 1. Excess consumption rate
        if "demand_volatility" in df.columns:
            df["excess_consumption_flag"] = df["demand_volatility"].fillna(0) > 1.5
        else:
            df["excess_consumption_flag"] = False

        # 2. Repeat failure rate
        if job_col in df.columns and part_col in df.columns:
            job_part_counts = df.groupby(job_col)[part_col].count().reset_index()
            job_part_counts.columns = [job_col, "parts_per_job"]
            df = pd.merge(df, job_part_counts, on=job_col, how="left")
            df["repeat_failure_flag"] = df["parts_per_job"].fillna(0) > 2
        else:
            df["repeat_failure_flag"] = False

        # 3. Warranty abuse score
        if process_col and process_col in df.columns:
            df["warranty_flag"] = df[process_col].astype(str).str.contains("AMC|WARRANTY", case=False, na=False)
        else:
            df["warranty_flag"] = False

        # 4. Stock mismatch (ordered != consumed)
        df["stock_mismatch_flag"] = abs(df.get("demand_gap", 0).fillna(0)) > 0

        # Aggregate by branch
        if branch_col and branch_col in df.columns:
            branch_leakage = df.groupby(branch_col).agg({
                "excess_consumption_flag": "mean",
                "repeat_failure_flag": "mean",
                "warranty_flag": "mean",
                "stock_mismatch_flag": "mean",
                "consumed_qty": "sum",
                job_col: "nunique",
            }).reset_index()

            branch_leakage.columns = [
                branch_col,
                "excess_consumption_rate",
                "repeat_failure_rate",
                "warranty_rate",
                "stock_mismatch_rate",
                "total_consumption",
                "unique_jobs",
            ]

            branch_leakage["revenue_leakage_score"] = (
                branch_leakage["excess_consumption_rate"] * 0.3
                + branch_leakage["repeat_failure_rate"] * 0.4
                + branch_leakage["warranty_rate"] * 0.2
                + branch_leakage["stock_mismatch_rate"] * 0.1
            )

            branch_leakage = branch_leakage.sort_values("revenue_leakage_score", ascending=False)
        else:
            branch_leakage = pd.DataFrame()

        # Aggregate by franchise
        if franchise_col and franchise_col in df.columns:
            franchise_leakage = df.groupby(franchise_col).agg({
                "excess_consumption_flag": "mean",
                "repeat_failure_flag": "mean",
                "warranty_flag": "mean",
                "stock_mismatch_flag": "mean",
                "consumed_qty": "sum",
                job_col: "nunique",
            }).reset_index()

            franchise_leakage.columns = [
                franchise_col,
                "excess_consumption_rate",
                "repeat_failure_rate",
                "warranty_rate",
                "stock_mismatch_rate",
                "total_consumption",
                "unique_jobs",
            ]

            franchise_leakage["revenue_leakage_score"] = (
                franchise_leakage["excess_consumption_rate"] * 0.3
                + franchise_leakage["repeat_failure_rate"] * 0.4
                + franchise_leakage["warranty_rate"] * 0.2
                + franchise_leakage["stock_mismatch_rate"] * 0.1
            )

            franchise_leakage = franchise_leakage.sort_values("revenue_leakage_score", ascending=False)
        else:
            franchise_leakage = pd.DataFrame()

        # High-risk spares
        if part_col and part_col in df.columns:
            spare_leakage = df.groupby(part_col).agg({
                "excess_consumption_flag": "mean",
                "repeat_failure_flag": "mean",
                "warranty_flag": "mean",
                "stock_mismatch_flag": "mean",
                "consumed_qty": "sum",
                job_col: "nunique",
            }).reset_index()

            spare_leakage.columns = [
                part_col,
                "excess_consumption_rate",
                "repeat_failure_rate",
                "warranty_rate",
                "stock_mismatch_rate",
                "total_consumption",
                "unique_jobs",
            ]

            spare_leakage["risk_score"] = (
                spare_leakage["excess_consumption_rate"] * 0.3
                + spare_leakage["repeat_failure_rate"] * 0.4
                + spare_leakage["warranty_rate"] * 0.2
                + spare_leakage["stock_mismatch_rate"] * 0.1
            )

            high_risk_spares = spare_leakage.nlargest(20, "risk_score")
        else:
            high_risk_spares = pd.DataFrame()

        logger.info("Revenue leakage detection completed")

        self.branch_leakage_summary = branch_leakage
        self.franchise_leakage_summary = franchise_leakage
        self.top_20_high_risk_spares = high_risk_spares

        return branch_leakage, franchise_leakage, high_risk_spares

    # ----------------------------
    # Full pipeline
    # ----------------------------
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete forecasting and leakage detection pipeline.
        """
        logger.info("=" * 80)
        logger.info("SPARE PARTS FORECASTING & REVENUE LEAKAGE DETECTION PIPELINE")
        logger.info("=" * 80)

        try:
            self.load_data()
            self.infer_canonical_fields()
            self.clean_and_validate_data()
            self.integrate_data()
            self.engineer_features()
            self.generate_demand_forecast()
            self.detect_revenue_leakage()

            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

            return {
                "normalized_clean_data": self.normalized_clean_data,
                "forecast_30_60_90": self.forecast_30_60_90,
                "branch_leakage_summary": self.branch_leakage_summary,
                "franchise_leakage_summary": self.franchise_leakage_summary,
                "top_20_high_risk_spares": self.top_20_high_risk_spares,
                "canonical_fields": self.canonical_fields,
                "join_loss_percentage": self.join_loss_percentage,
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise


# TODO: LLM explanation layer - generate natural language insights
