import streamlit as st
def display_title():
    st.markdown("""
    <div style='background-color:#00bfff; padding:10px; border-radius:10px; text-align:center;'>
        <h1 style='color:white;'>AI-Powered Strategic Navigator for Business</h1>
    </div>
    """, unsafe_allow_html=True)

def display_navigation():
    st.markdown("""
    <style>
    .nav-buttons {
        display: flex;
        justify-content: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .nav-button {
        margin-right: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    nav_items = [
        "🏠 Home",
        "🔍 Q&A System",
        "📊 Data Insights",
        "💡 AI-Powered Strategies",
        "📈 Metric Tracking",
        "💼 Company Analysis",
        "📊 Auto-Adaptive Business Strategy Maps",
        "💸 Revenue Leakage Analysis",
        "💬 Feedback"
    ]

    selected_page = None
    cols = st.columns(len(nav_items))
    for idx, (col, item) in enumerate(zip(cols, nav_items)):
        if col.button(item):
            selected_page = item
            st.session_state['page'] = selected_page
    if 'page' in st.session_state:
        selected_page = st.session_state['page']
    return selected_page

def display_footer():
    st.markdown("""
    <hr>
    <div style='text-align:center;'>
        <p style='color:gray;'>© 2024 AI-Powered Strategic Navigator for Business. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
