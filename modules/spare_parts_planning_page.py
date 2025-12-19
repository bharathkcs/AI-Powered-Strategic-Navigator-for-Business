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

        status_text.text("🧠 Generating decision intelligence insights...")
        progress_bar.progress(90)
        engine.generate_executive_insights()
        engine.generate_demand_insights()
        engine.generate_leakage_insights()
        engine.generate_spare_part_narratives()
        st.success("✅ Decision intelligence insights generated")

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")

        display_results(
            engine.normalized_clean_data,
            forecast_df,
            branch_leakage,
            franchise_leakage,
            high_risk_spares,
            engine.canonical_fields,
            engine.executive_insights,
            engine.demand_insights,
            engine.leakage_insights,
            engine.spare_part_narratives
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
    canonical_fields: dict,
    executive_insights: dict = None,
    demand_insights: pd.DataFrame = None,
    leakage_insights: pd.DataFrame = None,
    spare_part_narratives: list = None
):
    """Display analysis results with visualizations"""

    # Executive Summary Section (TOP OF PAGE)
    if executive_insights:
        st.header("🔑 Executive Summary")

        # System health badge
        health = executive_insights["system_health"]
        health_colors = {
            "Healthy": "🟢",
            "Warning": "🟡",
            "Critical": "🔴"
        }
        st.markdown(f"### System Health: {health_colors.get(health, '⚪')} {health}")

        # CXO one-liner
        st.info(f"**{executive_insights['one_line_cxo_summary']}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Trust Score", f"{executive_insights['data_trust_score']}%")
        with col2:
            st.metric("Forecast Confidence", executive_insights["forecast_confidence"])
        with col3:
            health_score = {"Healthy": 100, "Warning": 60, "Critical": 20}
            st.metric("System Health Score", health_score.get(health, 0))

        # Top 3 Risks
        st.markdown("#### ⚠️ Top 3 Risks")
        for i, risk in enumerate(executive_insights["top_3_risks"], 1):
            st.markdown(f"{i}. {risk}")

        # Top 3 Actions
        st.markdown("#### ✅ Top 3 Recommended Actions")
        for i, action in enumerate(executive_insights["top_3_actions"], 1):
            st.markdown(f"{i}. {action}")

        st.markdown("---")

    st.header("📊 Analysis Results")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Demand Forecasts",
        "💸 Revenue Leakage",
        "🔍 Data Quality",
        "📋 Detailed Data",
        "📥 Export Results"
    ])

    with tab1:
        st.subheader("📦 Demand Signals & Planning Actions")

        if demand_insights is not None and not demand_insights.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Parts Forecasted", demand_insights["part_id"].nunique())
            with col2:
                rising_count = (demand_insights["demand_signal"] == "Rising").sum()
                st.metric("Rising Demand Parts", rising_count)
            with col3:
                action_increase = (demand_insights["planning_action"] == "Increase Stock").sum()
                st.metric("Stock Increase Required", action_increase)

            st.markdown("### 📊 Demand Signal Overview")

            horizon = st.selectbox(
                "Select Forecast Horizon",
                ["30_day", "60_day", "90_day"],
                format_func=lambda x: x.replace("_", " ").title()
            )

            horizon_insights = demand_insights[demand_insights["forecast_horizon"] == horizon]

            if not horizon_insights.empty:
                # Signal distribution
                signal_counts = horizon_insights["demand_signal"].value_counts()
                col1, col2 = st.columns(2)

                with col1:
                    fig_signal = px.pie(
                        values=signal_counts.values,
                        names=signal_counts.index,
                        title="Demand Signal Distribution",
                        color=signal_counts.index,
                        color_discrete_map={
                            "Rising": "#00CC96",
                            "Stable": "#636EFA",
                            "Declining": "#EF553B"
                        }
                    )
                    st.plotly_chart(fig_signal, use_container_width=True)

                with col2:
                    action_counts = horizon_insights["planning_action"].value_counts()
                    fig_action = px.bar(
                        x=action_counts.index,
                        y=action_counts.values,
                        title="Recommended Planning Actions",
                        labels={"x": "Action", "y": "Count"},
                        color=action_counts.values
                    )
                    st.plotly_chart(fig_action, use_container_width=True)

                st.markdown("### 🎯 Actionable Demand Insights")

                # Filter options
                signal_filter = st.multiselect(
                    "Filter by Demand Signal",
                    ["Rising", "Stable", "Declining"],
                    default=["Rising", "Declining"]
                )

                filtered_insights = horizon_insights[horizon_insights["demand_signal"].isin(signal_filter)]

                # Display insights with color coding
                for _, row in filtered_insights.head(20).iterrows():
                    signal_emoji = {"Rising": "📈", "Stable": "➡️", "Declining": "📉"}
                    action_emoji = {"Increase Stock": "⬆️", "Maintain": "↔️", "Reduce": "⬇️", "Monitor Closely": "👀"}

                    with st.expander(f"{signal_emoji.get(row['demand_signal'], '')} {row['part_id']} - {row['planning_action']}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Demand Signal", row["demand_signal"])
                        with col2:
                            st.metric("Planning Action", row["planning_action"])
                        with col3:
                            st.metric("Confidence", row["forecast_confidence"])

                        st.markdown(f"**Forecast:** {row['forecast_demand']:.1f} units")
                        st.markdown(f"**Historical Average:** {row['historical_avg_demand']:.1f} units")
                        st.markdown(f"**Explanation:** {row['reason']}")

                with st.expander("📋 View All Forecast Data (Expandable)", expanded=False):
                    st.dataframe(
                        horizon_insights.sort_values("forecast_demand", ascending=False),
                        use_container_width=True
                    )
        elif forecast_df is not None and not forecast_df.empty:
            st.warning("⚠️ Demand insights not available. Showing raw forecast data.")
            st.dataframe(forecast_df, use_container_width=True)
        else:
            st.warning("No forecasts generated. Insufficient historical data.")

    with tab2:
        st.subheader("🚨 Service-Led Revenue Leakage Analysis with Explanations")

        if leakage_insights is not None and not leakage_insights.empty:
            st.markdown("### 🏢 Branch-Level Leakage with Root Cause Analysis")
            top_branches = leakage_insights.head(10)

            fig = px.bar(
                top_branches,
                x=leakage_insights.columns[0],
                y="revenue_leakage_score",
                title="Top 10 Branches by Revenue Leakage Score",
                labels={leakage_insights.columns[0]: "Branch", "revenue_leakage_score": "Leakage Score"},
                color="revenue_leakage_score",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 💡 Branch Leakage Insights")

            for _, branch_row in top_branches.iterrows():
                branch_id = branch_row[leakage_insights.columns[0]]
                score = branch_row["revenue_leakage_score"]

                score_color = "🔴" if score > 0.5 else "🟡" if score > 0.3 else "🟢"

                with st.expander(f"{score_color} Branch {branch_id} - Leakage Score: {score:.2f}"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Excess Consumption", f"{branch_row['excess_consumption_rate']*100:.1f}%")
                    with col2:
                        st.metric("Repeat Failures", f"{branch_row['repeat_failure_rate']*100:.1f}%")
                    with col3:
                        st.metric("Warranty Rate", f"{branch_row['warranty_rate']*100:.1f}%")
                    with col4:
                        st.metric("Stock Mismatch", f"{branch_row['stock_mismatch_rate']*100:.1f}%")

                    st.markdown("**🔍 Root Cause Analysis:**")
                    st.info(branch_row["root_cause_explanation"])

                    st.markdown("**🎯 Dominant Driver:**")
                    st.warning(f"{branch_row['dominant_leakage_driver']} (Secondary: {branch_row['secondary_driver']})")

                    st.markdown("**✅ Recommended Fix:**")
                    st.success(branch_row["recommended_fix"])

            with st.expander("📋 View Full Branch Leakage Data", expanded=False):
                st.dataframe(leakage_insights, use_container_width=True)

        elif not branch_leakage.empty:
            st.warning("⚠️ Leakage insights not available. Showing raw leakage data.")
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

        # High-Risk Spare Stories Section
        if spare_part_narratives is not None and len(spare_part_narratives) > 0:
            st.markdown("---")
            st.markdown("### ⚠️ High-Risk Spare Parts: Executive Briefing")
            st.markdown("**Consulting-style insights for immediate action**")

            for i, narrative in enumerate(spare_part_narratives, 1):
                st.markdown(f"**{i}.** {narrative}")
                st.markdown("")

        if not high_risk_spares.empty:
            st.markdown("---")
            st.markdown("### 📊 High-Risk Spares Visualization")

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
                },
                color="risk_score",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 View Full High-Risk Spares Data", expanded=False):
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

        # Export Executive Insights
        if executive_insights is not None:
            st.markdown("### 🔑 Executive Insights")
            insights_text = f"""
EXECUTIVE SUMMARY - SPARE PARTS ANALYSIS
{'='*60}

System Health: {executive_insights['system_health']}
Data Trust Score: {executive_insights['data_trust_score']}%
Forecast Confidence: {executive_insights['forecast_confidence']}

CXO Summary:
{executive_insights['one_line_cxo_summary']}

TOP 3 RISKS:
{chr(10).join(f'{i}. {risk}' for i, risk in enumerate(executive_insights['top_3_risks'], 1))}

TOP 3 RECOMMENDED ACTIONS:
{chr(10).join(f'{i}. {action}' for i, action in enumerate(executive_insights['top_3_actions'], 1))}
"""
            st.download_button(
                label="📥 Download Executive Summary (TXT)",
                data=insights_text,
                file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

        # Export Demand Insights
        if demand_insights is not None and not demand_insights.empty:
            st.markdown("### 📦 Demand Insights")
            csv_demand = demand_insights.to_csv(index=False)
            st.download_button(
                label="📥 Download Demand Insights (CSV)",
                data=csv_demand,
                file_name=f"demand_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        # Export Leakage Insights
        if leakage_insights is not None and not leakage_insights.empty:
            st.markdown("### 🚨 Leakage Insights")
            csv_leakage = leakage_insights.to_csv(index=False)
            st.download_button(
                label="📥 Download Leakage Insights (CSV)",
                data=csv_leakage,
                file_name=f"leakage_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        # Export Spare Part Narratives
        if spare_part_narratives is not None and len(spare_part_narratives) > 0:
            st.markdown("### ⚠️ High-Risk Spare Narratives")
            narratives_text = "\n\n".join(f"{i}. {narrative}" for i, narrative in enumerate(spare_part_narratives, 1))
            st.download_button(
                label="📥 Download Spare Part Narratives (TXT)",
                data=narratives_text,
                file_name=f"spare_part_narratives_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

        st.markdown("---")
        st.markdown("### 📦 Complete Analysis Package")

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Executive Insights
            if executive_insights is not None:
                exec_df = pd.DataFrame([{
                    "Metric": k,
                    "Value": str(v) if not isinstance(v, list) else "; ".join(v)
                } for k, v in executive_insights.items()])
                exec_df.to_excel(writer, sheet_name="Executive_Insights", index=False)

            # Demand Insights
            if demand_insights is not None and not demand_insights.empty:
                demand_insights.to_excel(writer, sheet_name="Demand_Insights", index=False)
            elif not forecast_df.empty:
                forecast_df.to_excel(writer, sheet_name="Forecasts", index=False)

            # Leakage Insights
            if leakage_insights is not None and not leakage_insights.empty:
                leakage_insights.to_excel(writer, sheet_name="Leakage_Insights", index=False)
            elif not branch_leakage.empty:
                branch_leakage.to_excel(writer, sheet_name="Branch_Leakage", index=False)

            if not franchise_leakage.empty:
                franchise_leakage.to_excel(writer, sheet_name="Franchise_Leakage", index=False)

            if not high_risk_spares.empty:
                high_risk_spares.to_excel(writer, sheet_name="High_Risk_Spares", index=False)

            # Spare Part Narratives
            if spare_part_narratives is not None and len(spare_part_narratives) > 0:
                narratives_df = pd.DataFrame({
                    "ID": range(1, len(spare_part_narratives) + 1),
                    "Narrative": spare_part_narratives
                })
                narratives_df.to_excel(writer, sheet_name="Spare_Narratives", index=False)

            schema_df = pd.DataFrame(
                [{"Canonical_Field": k, "Actual_Column": v} for k, v in canonical_fields.items()]
            )
            schema_df.to_excel(writer, sheet_name="Schema_Mapping", index=False)

        excel_bytes = output.getvalue()
        st.download_button(
            label="📥 Download Complete Decision Intelligence Package (Excel)",
            data=excel_bytes,
            file_name=f"decision_intelligence_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("✅ All decision intelligence results ready for export!")
