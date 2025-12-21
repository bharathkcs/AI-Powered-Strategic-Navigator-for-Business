"""
Spare Parts Planning - Streamlit Page (BUSINESS-GRADE VERSION)

This module provides INSIGHT-FIRST UI for spare parts demand forecasting
and service-led revenue leakage analysis.

DESIGN PHILOSOPHY:
- Purpose first: Every page section explains WHAT it is for
- Insights first: Business narratives before charts
- Actions first: Tell users WHAT TO DO before showing data
- Clarity first: Plain English, not technical jargon

This mirrors the gold standard approach from ifb_service_forecasting.py
and revenue_leakage_detector.py.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime

from modules.spare_parts_forecasting import SparePartsForecastingEngine, SchemaInferenceError


def render_spare_parts_planning_page():
    """Render the spare parts planning page with INSIGHT-FIRST design"""

    # ==============================================
    # PAGE PURPOSE & HOW TO USE (MANDATORY SECTION)
    # ==============================================
    st.title("🔧 IFB Spare Parts - Decision Intelligence System")

    st.info("""
    ### 📖 What This Page Does

    This system transforms your spare parts data into **actionable business intelligence**:

    ✅ **Demand Forecasting**: Predicts which parts you'll need in 30/60/90 days and tells you whether to increase, reduce, or maintain inventory

    ✅ **Revenue Leakage Detection**: Identifies where you're losing money through excess consumption, repeat failures, warranty abuse, or stock mismatches

    ✅ **Executive Insights**: Answers critical questions: Is your system healthy? What's broken? Why? What should you do next week?

    ### 👥 Who Should Use This

    - **Operations Leaders**: Get clear actions for inventory planning and leakage prevention
    - **Branch Managers**: Understand which parts are problematic and why
    - **CXOs**: See system health and top 3 risks/actions at a glance
    - **Procurement Teams**: Know exactly what to order and when

    ### 🎯 How to Read the Outputs

    1. **Executive Summary** (top): Start here - tells you if things are healthy/warning/critical
    2. **Demand Insights**: Shows which parts need more stock, which need less, and WHY
    3. **Revenue Leakage**: Explains which branches/parts are losing money and WHAT TO FIX
    4. **High-Risk Spares**: Bullet-point narratives of your most critical spare parts problems

    **You do NOT need to understand the math.** This system explains everything in plain English.
    """)

    st.markdown("---")

    # ==============================================
    # DATA UPLOAD SECTION
    # ==============================================
    st.header("📁 Step 1: Upload Your Data")

    st.info("""
    **Expected Excel File Structure:**
    - **Sheet 1: INDENT** - Spare parts orders/requests
    - **Sheet 2: SPARES_CONSUMED** - Actual parts used in service
    - **Sheet 3: BRANCHES** - Branch reference data
    - **Sheet 4: FRANCHISES** - Franchise partner data

    The system automatically adapts to your column names - no manual configuration needed.
    """)

    uploaded_file = st.file_uploader(
        "Upload Excel file with spare parts data",
        type=["xlsx", "xls"],
        help="File must contain 4 sheets: INDENT, SPARES_CONSUMED, BRANCHES, FRANCHISES"
    )

    if uploaded_file is None:
        st.warning("⬆️ Please upload an Excel file to begin analysis")
        st.stop()

    # ==============================================
    # DATA PROCESSING
    # ==============================================
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
                st.stop()

            for sheet in required_sheets:
                excel_data[sheet] = pd.read_excel(uploaded_file, sheet_name=sheet)

            st.success(f"✅ Successfully loaded {len(required_sheets)} sheets")

        # Initialize engine with LLM if available
        llm_instance = st.session_state.get('llm', None)

        with st.spinner("🔬 Initializing forecasting engine..."):
            engine = SparePartsForecastingEngine(excel_data=excel_data, llm=llm_instance)

        # ==============================================
        # RUN ANALYSIS BUTTON
        # ==============================================
        st.header("🚀 Step 2: Run Analysis")

        st.markdown("""
        Click below to run the complete analysis. This will:
        1. Validate your data quality
        2. Generate demand forecasts for all spare parts
        3. Detect revenue leakage patterns
        4. Create business-grade insights and recommendations

        **Time: 30-60 seconds** depending on data volume
        """)

        if st.button("▶️ Run Full Analysis & Generate Insights", type="primary", use_container_width=True):
            run_analysis_and_display(engine, llm_instance)

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        with st.expander("🔍 Technical Details"):
            st.exception(e)


def run_analysis_and_display(engine: SparePartsForecastingEngine, llm=None):
    """Execute analysis pipeline and display INSIGHT-FIRST results"""

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Pipeline execution (existing code, kept for backward compatibility)
        status_text.text("📂 Loading data...")
        progress_bar.progress(10)
        engine.load_data()

        status_text.text("🔍 Inferring column schema...")
        progress_bar.progress(20)
        try:
            canonical_fields = engine.infer_canonical_fields()
        except SchemaInferenceError as e:
            st.error("❌ Schema inference failed!")
            st.error(str(e))
            st.stop()

        status_text.text("🧹 Cleaning and validating data...")
        progress_bar.progress(30)
        engine.clean_and_validate_data()

        status_text.text("🔗 Integrating datasets...")
        progress_bar.progress(40)
        integrated_data = engine.integrate_data()

        status_text.text("⚙️ Engineering features...")
        progress_bar.progress(55)
        engine.engineer_features()

        status_text.text("📈 Generating demand forecasts...")
        progress_bar.progress(70)
        forecast_df = engine.generate_demand_forecast()

        status_text.text("💸 Detecting revenue leakage...")
        progress_bar.progress(85)
        branch_leakage, franchise_leakage, high_risk_spares = engine.detect_revenue_leakage()

        status_text.text("🧠 Generating decision intelligence insights...")
        progress_bar.progress(95)

        # Generate executive summary (with or without LLM)
        engine.generate_executive_summary()

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")

        st.success("✅ **Analysis Complete!** Scroll down to see your business insights.")
        st.markdown("---")

        # ==============================================
        # DISPLAY INSIGHT-FIRST RESULTS
        # ==============================================
        display_business_intelligence(
            engine=engine,
            clean_data=engine.normalized_clean_data,
            forecast_df=engine.forecast_30_60_90,
            branch_leakage=engine.branch_leakage_summary,
            franchise_leakage=engine.franchise_leakage_summary,
            high_risk_spares=engine.top_20_high_risk_spares,
            canonical_fields=engine.canonical_fields,
            executive_summary=engine.executive_summary,
            demand_insights=engine.demand_insights,
            leakage_insights=engine.leakage_insights,
            spare_part_insights=engine.spare_part_insights
        )

    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        with st.expander("🔍 Technical Details"):
            st.exception(e)


def display_business_intelligence(
    engine,
    clean_data: pd.DataFrame,
    forecast_df: pd.DataFrame,
    branch_leakage: pd.DataFrame,
    franchise_leakage: pd.DataFrame,
    high_risk_spares: pd.DataFrame,
    canonical_fields: dict,
    executive_summary: dict = None,
    demand_insights: pd.DataFrame = None,
    leakage_insights: pd.DataFrame = None,
    spare_part_insights: list = None
):
    """
    Display results in INSIGHT-FIRST format.

    Structure:
    1. Executive Summary (TOP - most important)
    2. Demand Insights (business explanations, not tables)
    3. Revenue Leakage Insights (narratives, not metrics)
    4. High-Risk Spare Parts (consulting-style stories)
    5. Supporting Data (hidden by default)
    """

    # ==============================================
    # 1. EXECUTIVE SUMMARY (MANDATORY TOP SECTION)
    # ==============================================
    st.header("📊 Executive Summary")

    st.markdown("""
    **This section answers:**
    - Is my spare parts system healthy, or is there a problem?
    - What are the top 3 biggest risks right now?
    - What 3 things should I do in the next 30 days?
    - Can I trust this forecast?
    """)

    if executive_summary:
        health = executive_summary["system_health"]
        health_colors = {"Healthy": "🟢", "Warning": "🟡", "Critical": "🔴"}
        health_emoji = health_colors.get(health, "⚪")

        # System Health Badge
        if health == "Healthy":
            st.success(f"### {health_emoji} System Status: {health}")
        elif health == "Warning":
            st.warning(f"### {health_emoji} System Status: {health}")
        else:
            st.error(f"### {health_emoji} System Status: {health}")

        # AI-generated narrative (if LLM available)
        if "ai_executive_narrative" in executive_summary:
            ai_narrative = executive_summary["ai_executive_narrative"]
            if ai_narrative and "not available" not in ai_narrative.lower() and "not configured" not in ai_narrative.lower():
                st.info(f"**🤖 AI Executive Summary:**\n\n{ai_narrative}")
            else:
                st.info("**📊 Executive Summary:** System has generated actionable insights based on data analysis. See risks and actions below for specific guidance.")

        # Key Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            trust_score = executive_summary['data_trust_score']
            if trust_score >= 80:
                col1.metric("📊 Data Trust Score", f"{trust_score}%", help="How reliable is this forecast?")
            elif trust_score >= 60:
                col1.metric("📊 Data Trust Score", f"{trust_score}%", delta="⚠️ Medium Confidence", help="Forecast is usable but has data quality issues")
            else:
                col1.metric("📊 Data Trust Score", f"{trust_score}%", delta="❌ Low Confidence", delta_color="inverse", help="Data quality issues compromise forecast reliability")

        with col2:
            confidence = executive_summary["forecast_confidence"]
            confidence_emoji = {"High": "✅", "Medium": "⚠️", "Low": "❌"}
            col2.metric("🔮 Forecast Confidence", f"{confidence_emoji.get(confidence, '')} {confidence}", help="How accurate are the forecasts?")

        with col3:
            health_score_map = {"Healthy": "95%", "Warning": "65%", "Critical": "30%"}
            col3.metric("🏥 System Health", health_score_map.get(health, "N/A"), help="Overall spare parts system performance")

        # Top 3 Risks
        st.markdown("### ⚠️ Top 3 Business Risks")
        st.markdown("**These are the biggest problems threatening your operations right now:**")
        for i, risk in enumerate(executive_summary["top_3_risks"], 1):
            st.warning(f"**{i}.** {risk}")

        # Top 3 Actions
        st.markdown("### ✅ Top 3 Recommended Actions (Next 30 Days)")
        st.markdown("**Do these things first to fix the biggest problems:**")
        for i, action in enumerate(executive_summary["top_3_actions"], 1):
            st.success(f"**{i}.** {action}")

    else:
        st.info("ℹ️ Executive summary generation in progress. Please ensure the analysis pipeline completed successfully.")

    st.markdown("---")

    # ==============================================
    # 2. DEMAND INSIGHTS (BUSINESS EXPLANATIONS)
    # ==============================================
    st.header("📦 Demand Forecasting Insights")

    st.markdown("""
    **This section answers:**
    - Which parts should I order more of?
    - Which parts should I reduce?
    - Why is demand changing?
    - How confident should I be in these forecasts?
    """)

    if demand_insights is not None and not demand_insights.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Parts Analyzed", demand_insights["part_id"].nunique())
        with col2:
            rising_count = (demand_insights["demand_signal"] == "Rising").sum()
            st.metric("📈 Rising Demand", rising_count)
        with col3:
            declining_count = (demand_insights["demand_signal"] == "Declining").sum()
            st.metric("📉 Declining Demand", declining_count)
        with col4:
            action_increase = (demand_insights["planning_action"] == "Increase Stock").sum()
            st.metric("⬆️ Increase Stock Needed", action_increase)

        # Horizon selector
        st.markdown("### 🔮 Forecast Horizon")
        horizon = st.selectbox(
            "Select time period for detailed insights:",
            ["30_day", "60_day", "90_day"],
            format_func=lambda x: x.replace("_", "-").replace("day", "Day Forecast").title(),
            help="Choose how far ahead you want to plan"
        )

        horizon_insights = demand_insights[demand_insights["forecast_horizon"] == horizon]

        if not horizon_insights.empty:
            # Filter controls
            st.markdown("### 🎯 Focus On:")
            signal_filter = st.multiselect(
                "Show parts with demand signal:",
                ["Rising", "Stable", "Declining"],
                default=["Rising", "Declining"],
                help="Focus on parts that need inventory adjustments"
            )

            filtered_insights = horizon_insights[horizon_insights["demand_signal"].isin(signal_filter)]

            if filtered_insights.empty:
                st.info("No parts match your filters. Adjust the selection above.")
            else:
                st.markdown(f"### 📋 Demand Insights for {len(filtered_insights)} Parts")
                st.markdown("**Read these to understand WHAT TO DO and WHY:**")

                # Display insights as business narratives
                for idx, row in filtered_insights.head(20).iterrows():
                    signal_emoji = {"Rising": "📈", "Stable": "➡️", "Declining": "📉"}
                    action_color = {
                        "Increase Stock": "success",
                        "Maintain Current Levels": "info",
                        "Reduce Stock": "warning",
                        "Monitor Closely": "error"
                    }

                    with st.expander(f"{signal_emoji.get(row['demand_signal'], '')} **Part {row['part_id']}** → {row['planning_action']}"):
                        # Action box
                        action_box_color = action_color.get(row['planning_action'], 'info')
                        if action_box_color == "success":
                            st.success(f"**Recommended Action:** {row['planning_action']}")
                        elif action_box_color == "warning":
                            st.warning(f"**Recommended Action:** {row['planning_action']}")
                        elif action_box_color == "error":
                            st.error(f"**Recommended Action:** {row['planning_action']}")
                        else:
                            st.info(f"**Recommended Action:** {row['planning_action']}")

                        # Business explanation (plain English)
                        st.markdown(f"**Why:** {row['business_explanation']}")

                        # Supporting metrics (collapsed by default)
                        with st.expander("📊 Supporting Numbers"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Forecasted Demand", f"{row['forecast_demand']:.1f} units/month")
                            with col2:
                                st.metric("Historical Average", f"{row['historical_avg_demand']:.1f} units/month")
                            with col3:
                                st.metric("Confidence Level", row["forecast_confidence"])

                st.markdown("---")
                st.info("💡 **Tip:** Focus on 'Increase Stock' and 'Reduce Stock' recommendations first. These have the biggest operational impact.")

        with st.expander("📋 View Complete Forecast Table (Expandable)"):
            st.dataframe(
                horizon_insights[[
                    "part_id", "demand_signal", "planning_action",
                    "forecast_demand", "historical_avg_demand",
                    "forecast_confidence", "business_explanation"
                ]].sort_values("forecast_demand", ascending=False),
                use_container_width=True
            )

    elif forecast_df is not None and not forecast_df.empty:
        # Fallback to raw forecast data (should not happen with new logic, but kept for safety)
        st.info("ℹ️ Showing forecast data. Demand insights generation in progress...")
        with st.expander("📊 Forecast Data"):
            st.dataframe(forecast_df, use_container_width=True)
    else:
        st.error("❌ No demand forecasts generated. This usually means insufficient historical data (need at least 3 months).")

    st.markdown("---")

    # ==============================================
    # 3. REVENUE LEAKAGE INSIGHTS (NARRATIVES)
    # ==============================================
    st.header("💸 Revenue Leakage Analysis")

    st.markdown("""
    **This section answers:**
    - Which branches are losing money and why?
    - What's causing the leakage (excess usage? repeat failures? warranty abuse?)?
    - What should I fix first?
    - What exact steps should each branch take?
    """)

    if leakage_insights is not None and not leakage_insights.empty:
        st.markdown("### 🏢 Branch-Level Leakage Insights")

        # Top branches visualization
        top_branches = leakage_insights.head(10)

        fig = px.bar(
            top_branches,
            x=leakage_insights.columns[0],
            y="revenue_leakage_score",
            title="Top 10 Branches by Revenue Leakage Score (Higher = Worse)",
            labels={leakage_insights.columns[0]: "Branch", "revenue_leakage_score": "Leakage Score"},
            color="revenue_leakage_score",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Business narratives for each branch
        st.markdown("### 💡 Branch-by-Branch Action Plan")
        st.markdown("**Read these to understand WHAT IS WRONG and WHAT TO FIX:**")

        for idx, branch_row in top_branches.iterrows():
            branch_id = branch_row[leakage_insights.columns[0]]
            score = branch_row["revenue_leakage_score"]

            # Color-code severity
            if score > 0.5:
                severity_emoji = "🔴"
                severity_text = "CRITICAL"
            elif score > 0.3:
                severity_emoji = "🟡"
                severity_text = "HIGH PRIORITY"
            else:
                severity_emoji = "🟢"
                severity_text = "ROUTINE"

            with st.expander(f"{severity_emoji} **Branch {branch_id}** — Leakage Score: {score:.2f} ({severity_text})"):
                # Root cause explanation
                st.markdown("### 🔍 What Is Going Wrong")
                st.error(branch_row["root_cause_explanation"])

                # Leakage drivers breakdown
                st.markdown("### 📊 Leakage Breakdown")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Excess Consumption", f"{branch_row['excess_consumption_rate']*100:.1f}%")
                with col2:
                    st.metric("Repeat Failures", f"{branch_row['repeat_failure_rate']*100:.1f}%")
                with col3:
                    st.metric("Warranty Rate", f"{branch_row['warranty_rate']*100:.1f}%")
                with col4:
                    st.metric("Stock Mismatch", f"{branch_row['stock_mismatch_rate']*100:.1f}%")

                # Dominant and secondary drivers
                st.markdown("### 🎯 Primary Issues")
                st.warning(f"**Main Problem:** {branch_row['dominant_leakage_driver']}")
                if branch_row['secondary_driver'] != "None":
                    st.info(f"**Contributing Factor:** {branch_row['secondary_driver']}")

                # Recommended fix
                st.markdown("### ✅ What To Do Next")
                st.success(branch_row["recommended_fix"])

        with st.expander("📋 View Complete Branch Leakage Data"):
            st.dataframe(leakage_insights, use_container_width=True)

    elif not branch_leakage.empty:
        # Fallback to raw data (should not happen with new logic, but kept for safety)
        st.info("ℹ️ Showing branch leakage data. Insights generation in progress...")
        with st.expander("📊 Branch Leakage Data"):
            st.dataframe(branch_leakage, use_container_width=True)
    else:
        st.info("ℹ️ No branch-level leakage data available.")

    # Franchise leakage (if available)
    if not franchise_leakage.empty:
        st.markdown("---")
        st.markdown("### 🏪 Franchise-Level Leakage")
        with st.expander("📊 View Franchise Leakage Data"):
            st.dataframe(franchise_leakage, use_container_width=True)

    st.markdown("---")

    # ==============================================
    # 4. HIGH-RISK SPARE PARTS (CONSULTING STORIES)
    # ==============================================
    st.header("⚠️ High-Risk Spare Parts Executive Briefing")

    st.markdown("""
    **This section answers:**
    - Which specific spare parts are causing the most problems?
    - Why are they problematic (quality issues? misuse? forecasting errors?)?
    - What business impact is this creating?
    - What immediate action should I take for each part?
    """)

    if spare_part_insights is not None and len(spare_part_insights) > 0:
        st.markdown("**Read these consulting-style narratives to understand your critical spare parts issues:**")

        for i, narrative in enumerate(spare_part_insights, 1):
            # Parse severity from narrative
            if "CRITICAL" in narrative:
                st.error(f"**{i}.** {narrative}")
            elif "HIGH" in narrative:
                st.warning(f"**{i}.** {narrative}")
            else:
                st.info(f"**{i}.** {narrative}")
            st.markdown("")

        st.markdown("---")
        st.success("✅ **Action Required:** Review these narratives with your operations and procurement teams within 7 days.")

    elif not high_risk_spares.empty:
        # Fallback to visualization (should not happen with new logic, but kept for safety)
        st.info("ℹ️ Showing high-risk spares data. Narrative insights generation in progress...")

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

        with st.expander("📋 View High-Risk Spares Data"):
            st.dataframe(high_risk_spares, use_container_width=True)
    else:
        st.info("ℹ️ No high-risk spare parts identified.")

    st.markdown("---")

    # ==============================================
    # 5. SUPPORTING DATA (HIDDEN BY DEFAULT)
    # ==============================================
    st.header("📋 Supporting Data & Technical Details")

    st.markdown("""
    **This section contains:**
    - Data quality reports
    - Raw forecast tables
    - Technical schema mappings
    - Complete datasets for export

    **Note:** These are for technical users and auditing. Business users should focus on the insights above.
    """)

    with st.expander("🔍 Data Quality Report"):
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

                st.metric("Total Records", f"{total_records:,}")
                st.metric("Clean Records", f"{clean_records:,}")
                st.metric("Quality Rate", f"{(clean_records / total_records * 100):.1f}%")

            st.markdown("### 🔍 Quality Issues")
            quality_table = pd.DataFrame({
                "Issue Type": quality_dist.index,
                "Count": quality_dist.values,
                "Percentage": (quality_dist.values / len(clean_data) * 100).round(2)
            })
            st.dataframe(quality_table, use_container_width=True)
        else:
            st.info("Data quality metrics not available.")

    with st.expander("🗂️ Schema Mapping (Column Detection)"):
        if canonical_fields:
            schema_df = pd.DataFrame(
                [{"Canonical Field": k, "Detected Column": v} for k, v in canonical_fields.items()]
            )
            st.dataframe(schema_df, use_container_width=True)
        else:
            st.info("Schema mapping not available.")

    with st.expander("📊 Raw Data Tables"):
        st.markdown("### Normalized Clean Data")
        if clean_data is not None and not clean_data.empty:
            show_rows = st.slider("Number of rows to display", 10, 500, 100)
            st.dataframe(clean_data.head(show_rows), use_container_width=True)
        else:
            st.info("No data available.")

    # ==============================================
    # 6. EXPORT & DOWNLOAD
    # ==============================================
    st.markdown("---")
    st.header("📥 Export Decision Intelligence Package")

    st.markdown("""
    Download all insights, forecasts, and recommendations in Excel format.
    Perfect for sharing with stakeholders or importing into other systems.
    """)

    # Generate Excel export
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Executive Summary
        if executive_summary is not None:
            exec_df = pd.DataFrame([{
                "Metric": k,
                "Value": str(v) if not isinstance(v, list) else "; ".join(v)
            } for k, v in executive_summary.items()])
            exec_df.to_excel(writer, sheet_name="Executive_Summary", index=False)

        # Demand Insights
        if demand_insights is not None and not demand_insights.empty:
            demand_insights.to_excel(writer, sheet_name="Demand_Insights", index=False)
        elif forecast_df is not None and not forecast_df.empty:
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
        if spare_part_insights is not None and len(spare_part_insights) > 0:
            narratives_df = pd.DataFrame({
                "ID": range(1, len(spare_part_insights) + 1),
                "Spare_Part_Insight": spare_part_insights
            })
            narratives_df.to_excel(writer, sheet_name="Spare_Part_Insights", index=False)

        # Schema Mapping
        if canonical_fields:
            schema_df = pd.DataFrame(
                [{"Canonical_Field": k, "Detected_Column": v} for k, v in canonical_fields.items()]
            )
            schema_df.to_excel(writer, sheet_name="Schema_Mapping", index=False)

    excel_bytes = output.getvalue()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download Complete Decision Intelligence Package (Excel)",
            data=excel_bytes,
            file_name=f"IFB_Decision_Intelligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col2:
        # Text summary export
        if executive_summary is not None:
            summary_text = f"""
IFB SPARE PARTS - EXECUTIVE SUMMARY
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM HEALTH: {executive_summary['system_health']}
DATA TRUST SCORE: {executive_summary['data_trust_score']}%
FORECAST CONFIDENCE: {executive_summary['forecast_confidence']}

AI EXECUTIVE NARRATIVE:
{executive_summary.get('ai_executive_narrative', 'Not available')}

TOP 3 RISKS:
{chr(10).join(f'{i}. {risk}' for i, risk in enumerate(executive_summary['top_3_risks'], 1))}

TOP 3 RECOMMENDED ACTIONS:
{chr(10).join(f'{i}. {action}' for i, action in enumerate(executive_summary['top_3_actions'], 1))}

{'='*60}
"""
            st.download_button(
                label="📄 Download Executive Summary (Text)",
                data=summary_text,
                file_name=f"Executive_Summary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    st.success("✅ **All decision intelligence outputs ready for export and stakeholder distribution!**")

    # Footer
    st.markdown("---")
    st.caption("IFB Industries — Spare Parts Decision Intelligence System | Powered by AI-Driven Analytics")
