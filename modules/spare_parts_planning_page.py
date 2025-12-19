"""
Spare Parts Planning - Streamlit Page

This module provides the UI for spare parts demand forecasting and
service-led revenue leakage analysis.

This is a standalone add-on page that does NOT modify existing functionality.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime

from modules.spare_parts_forecasting import SparePartsForecastingEngine, SchemaInferenceError


def render_spare_parts_planning_page():
    """Render the spare parts planning page"""

    st.title("🔧 Spare Parts Demand Forecasting & Revenue Leakage Analysis")
    st.markdown("""
    **Enterprise Decision Intelligence for IFB Industries**

    This module provides:
    - **Schema-driven analysis** - Works with changing column names
    - **Demand forecasting** - 30/60/90-day predictions for spare parts
    - **Revenue leakage detection** - Identify service-led inefficiencies
    - **Quality insights** - Data validation and anomaly detection
    """)

    st.header("📁 Data Upload")

    st.info("""
    **Expected Excel File Structure:**
    - **Sheet 1: INDENT** - Spare parts demand/request data
    - **Sheet 2: SPARES_CONSUMED** - Actual spare parts consumption
    - **Sheet 3: BRANCHES** - Branch reference data
    - **Sheet 4: FRANCHISES** - Franchise reference data
    """)

    uploaded_file = st.file_uploader(
        "Upload Excel file with spare parts data",
        type=["xlsx", "xls"],
        help="File must contain 4 sheets: INDENT, SPARES_CONSUMED, BRANCHES, FRANCHISES"
    )

    if uploaded_file is None:
        st.warning("⬆️ Please upload an Excel file to begin analysis")
        return

    try:
        with st.spinner("🔄 Loading and processing data..."):
            excel_data = {}
            excel_file = pd.ExcelFile(uploaded_file)

            required_sheets = ["INDENT", "SPARES_CONSUMED", "BRANCHES", "FRANCHISES"]
            available_sheets = excel_file.sheet_names

            missing_sheets = [s for s in required_sheets if s not in available_sheets]
            if missing_sheets:
                st.error(f"❌ Missing required sheets: {missing_sheets}")
                st.info(f"Available sheets: {available_sheets}")
                return

            for sheet in required_sheets:
                excel_data[sheet] = pd.read_excel(uploaded_file, sheet_name=sheet)

            st.success(f"✅ Successfully loaded {len(required_sheets)} sheets")

            with st.expander("📊 Data Overview", expanded=False):
                for sheet_name, df in excel_data.items():
                    st.subheader(sheet_name)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rows", len(df))
                    with col2:
                        st.metric("Columns", len(df.columns))
                    st.caption(
                        f"Columns: {', '.join(df.columns[:10])}"
                        f"{'...' if len(df.columns) > 10 else ''}"
                    )

        with st.spinner("🔬 Initializing forecasting engine..."):
            engine = SparePartsForecastingEngine(excel_data=excel_data)

        st.header("🚀 Analysis Pipeline")

        if st.button("▶️ Run Full Analysis", type="primary", use_container_width=True):
            run_analysis(engine)

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)


def run_analysis(engine: SparePartsForecastingEngine):
    """Execute the full analysis pipeline and display results"""

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("📂 Loading data...")
        progress_bar.progress(10)
        engine.load_data()
        st.success("✅ Data loaded successfully")

        status_text.text("🔍 Inferring column schema...")
        progress_bar.progress(20)
        try:
            canonical_fields = engine.infer_canonical_fields()
            st.success(f"✅ Inferred {len(canonical_fields)} canonical fields")

            with st.expander("🗂️ Schema Mapping", expanded=False):
                schema_df = pd.DataFrame(
                    [{"Canonical Field": k, "Actual Column": v} for k, v in canonical_fields.items()]
                )
                st.dataframe(schema_df, use_container_width=True)

        except SchemaInferenceError as e:
            st.error("❌ Schema inference failed!")
            st.error(str(e))
            st.stop()

        status_text.text("🧹 Cleaning and validating data...")
        progress_bar.progress(30)
        engine.clean_and_validate_data()
        st.success("✅ Data cleaned and validated")

        status_text.text("🔗 Integrating datasets...")
        progress_bar.progress(40)
        integrated_data = engine.integrate_data()
        st.success(f"✅ Integrated {len(integrated_data)} records")
        st.info(f"Join loss: {engine.join_loss_percentage:.2f}%")

        status_text.text("⚙️ Engineering features...")
        progress_bar.progress(55)
        engine.engineer_features()
        st.success("✅ Features engineered")

        status_text.text("📈 Generating demand forecasts...")
        progress_bar.progress(70)
        forecast_df = engine.generate_demand_forecast()
        st.success(f"✅ Generated forecasts for {len(forecast_df)} combinations")

        status_text.text("💸 Detecting revenue leakage...")
        progress_bar.progress(85)
        branch_leakage, franchise_leakage, high_risk_spares = engine.detect_revenue_leakage()
        st.success("✅ Revenue leakage analysis completed")

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")

        display_results(
            engine.normalized_clean_data,
            forecast_df,
            branch_leakage,
            franchise_leakage,
            high_risk_spares,
            engine.canonical_fields
        )

    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)


def display_results(
    clean_data: pd.DataFrame,
    forecast_df: pd.DataFrame,
    branch_leakage: pd.DataFrame,
    franchise_leakage: pd.DataFrame,
    high_risk_spares: pd.DataFrame,
    canonical_fields: dict
):
    """Display analysis results with visualizations"""

    st.header("📊 Analysis Results")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Demand Forecasts",
        "💸 Revenue Leakage",
        "🔍 Data Quality",
        "📋 Detailed Data",
        "📥 Export Results"
    ])

    with tab1:
        st.subheader("Spare Parts Demand Forecasts")

        if forecast_df.empty:
            st.warning("No forecasts generated. Insufficient historical data.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Parts Forecasted", forecast_df["part_id"].nunique())
            with col2:
                st.metric("Total Forecast Horizons", len(forecast_df))
            with col3:
                avg_demand = forecast_df["forecast_demand"].mean()
                st.metric("Avg Forecasted Demand", f"{avg_demand:.2f}")

            st.markdown("### 📅 Forecast Analysis")

            horizon = st.selectbox(
                "Select Forecast Horizon",
                ["30_day", "60_day", "90_day"],
                format_func=lambda x: x.replace("_", " ").title()
            )

            forecast_horizon_df = forecast_df[forecast_df["forecast_horizon"] == horizon]

            if not forecast_horizon_df.empty:
                top_parts = forecast_horizon_df.nlargest(10, "forecast_demand")

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_parts["part_id"],
                    y=top_parts["forecast_demand"],
                    name="Forecast"
                ))
                fig.add_trace(go.Scatter(
                    x=top_parts["part_id"],
                    y=top_parts["ci_upper"],
                    mode="lines",
                    name="Upper CI",
                    line=dict(dash="dash")
                ))
                fig.add_trace(go.Scatter(
                    x=top_parts["part_id"],
                    y=top_parts["ci_lower"],
                    mode="lines",
                    name="Lower CI",
                    line=dict(dash="dash"),
                    fill="tonexty"
                ))

                fig.update_layout(
                    title=f"Top 10 Parts - {horizon.replace('_', ' ').title()} Forecast",
                    xaxis_title="Part ID",
                    yaxis_title="Forecasted Demand",
                    hovermode="x unified",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📋 Detailed Forecasts")
                st.dataframe(
                    forecast_horizon_df.sort_values("forecast_demand", ascending=False),
                    use_container_width=True
                )

    with tab2:
        st.subheader("Service-Led Revenue Leakage Analysis")

        if not branch_leakage.empty:
            st.markdown("### 🏢 Branch-Level Leakage")
            top_branches = branch_leakage.head(10)

            fig = px.bar(
                top_branches,
                x=branch_leakage.columns[0],
                y="revenue_leakage_score",
                title="Top 10 Branches by Revenue Leakage Score",
                labels={branch_leakage.columns[0]: "Branch", "revenue_leakage_score": "Leakage Score"},
                color="revenue_leakage_score"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📊 Leakage Components Breakdown")
            selected_branch = st.selectbox(
                "Select Branch for Details",
                top_branches[branch_leakage.columns[0]].tolist()
            )
            branch_data = top_branches[top_branches[branch_leakage.columns[0]] == selected_branch].iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Excess Consumption", f"{branch_data['excess_consumption_rate']*100:.1f}%")
            with c2:
                st.metric("Repeat Failures", f"{branch_data['repeat_failure_rate']*100:.1f}%")
            with c3:
                st.metric("Warranty Rate", f"{branch_data['warranty_rate']*100:.1f}%")
            with c4:
                st.metric("Stock Mismatch", f"{branch_data['stock_mismatch_rate']*100:.1f}%")

            st.dataframe(branch_leakage, use_container_width=True)

        if not franchise_leakage.empty:
            st.markdown("### 🏪 Franchise-Level Leakage")
            top_franchises = franchise_leakage.head(10)

            fig = px.bar(
                top_franchises,
                x=franchise_leakage.columns[0],
                y="revenue_leakage_score",
                title="Top 10 Franchises by Revenue Leakage Score",
                labels={franchise_leakage.columns[0]: "Franchise", "revenue_leakage_score": "Leakage Score"},
                color="revenue_leakage_score"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(franchise_leakage, use_container_width=True)

        if not high_risk_spares.empty:
            st.markdown("### ⚠️ Top 20 High-Risk Spare Parts")

            fig = px.scatter(
                high_risk_spares,
                x="total_consumption",
                y="risk_score",
                size="unique_jobs",
                hover_data=[high_risk_spares.columns[0]],
                title="High-Risk Spares: Consumption vs Risk Score",
                labels={
                    "total_consumption": "Total Consumption",
                    "risk_score": "Risk Score",
                    "unique_jobs": "Unique Jobs"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(high_risk_spares, use_container_width=True)

    with tab3:
        st.subheader("Data Quality Analysis")

        if clean_data is not None and not clean_data.empty and "data_quality_flag" in clean_data.columns:
            quality_dist = clean_data["data_quality_flag"].value_counts()

            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(
                    values=quality_dist.values,
                    names=quality_dist.index,
                    title="Data Quality Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                total_records = len(clean_data)
                clean_records = (clean_data["data_quality_flag"] == "clean").sum()

                st.metric("Total Records", total_records)
                st.metric("Clean Records", clean_records)
                st.metric("Quality Rate", f"{(clean_records / total_records * 100):.2f}%")

            st.markdown("### 🔍 Quality Issues")
            quality_table = pd.DataFrame({
                "Issue Type": quality_dist.index,
                "Count": quality_dist.values,
                "Percentage": (quality_dist.values / len(clean_data) * 100).round(2)
            })
            st.dataframe(quality_table, use_container_width=True)

    with tab4:
        st.subheader("Detailed Data View")

        st.markdown("### 🗂️ Normalized Clean Data")
        st.caption(f"Total records: {len(clean_data)}")

        col1, col2 = st.columns(2)
        with col1:
            if "data_quality_flag" in clean_data.columns:
                quality_filter = st.multiselect(
                    "Filter by Quality",
                    clean_data["data_quality_flag"].unique(),
                    default=["clean"]
                )
                filtered_data = clean_data[clean_data["data_quality_flag"].isin(quality_filter)]
            else:
                filtered_data = clean_data

        with col2:
            show_rows = st.slider("Number of rows to display", 10, 1000, 100)

        st.dataframe(filtered_data.head(show_rows), use_container_width=True)

        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"spare_parts_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with tab5:
        st.subheader("Export Analysis Results")

        if not forecast_df.empty:
            st.markdown("### 📈 Demand Forecasts")
            csv_forecast = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecasts (CSV)",
                data=csv_forecast,
                file_name=f"demand_forecasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            if not branch_leakage.empty:
                csv_branch = branch_leakage.to_csv(index=False)
                st.download_button(
                    label="📥 Download Branch Leakage (CSV)",
                    data=csv_branch,
                    file_name=f"branch_leakage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        with col2:
            if not franchise_leakage.empty:
                csv_franchise = franchise_leakage.to_csv(index=False)
                st.download_button(
                    label="📥 Download Franchise Leakage (CSV)",
                    data=csv_franchise,
                    file_name=f"franchise_leakage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        with col3:
            if not high_risk_spares.empty:
                csv_risk = high_risk_spares.to_csv(index=False)
                st.download_button(
                    label="📥 Download High-Risk Spares (CSV)",
                    data=csv_risk,
                    file_name=f"high_risk_spares_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        st.markdown("### 📦 Complete Analysis Package")

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            if not forecast_df.empty:
                forecast_df.to_excel(writer, sheet_name="Forecasts", index=False)
            if not branch_leakage.empty:
                branch_leakage.to_excel(writer, sheet_name="Branch_Leakage", index=False)
            if not franchise_leakage.empty:
                franchise_leakage.to_excel(writer, sheet_name="Franchise_Leakage", index=False)
            if not high_risk_spares.empty:
                high_risk_spares.to_excel(writer, sheet_name="High_Risk_Spares", index=False)

            schema_df = pd.DataFrame(
                [{"Canonical_Field": k, "Actual_Column": v} for k, v in canonical_fields.items()]
            )
            schema_df.to_excel(writer, sheet_name="Schema_Mapping", index=False)

        excel_bytes = output.getvalue()
        st.download_button(
            label="📥 Download Complete Analysis (Excel)",
            data=excel_bytes,
            file_name=f"spare_parts_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("✅ All results ready for export!")
