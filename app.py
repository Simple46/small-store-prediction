import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Sales Revenue Predictor",
                   page_icon="📈", layout="wide")

# ==========================================
# 2. LOAD DATA & MODEL (With Caching)
# ==========================================


@st.cache_data
def load_data():
    try:
        # Load the daily dataset for the EDA page
        df = pd.read_csv('daily_store_sales.csv', parse_dates=['Order Date'])
        return df
    except FileNotFoundError:
        st.error(
            "❌ Could not find 'daily_store_sales.csv'. Make sure it's in the same folder as this app.")
        st.stop()


@st.cache_resource
def load_model():
    try:
        # Load the trained model and scaler for the AI page
        model = joblib.load('random_forest_monthly_model.pkl')
        scaler = joblib.load('monthly_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error(
            "❌ Could not find the model or scaler files. Ensure the .pkl files are in this folder.")
        st.stop()


df = load_data()
model, scaler = load_model()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select a Page", ["📊 EDA Dashboard", "🤖 AI Predictor"])

# ==========================================
# 4. PAGE 1: EDA DASHBOARD
# ==========================================
if page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")
    st.markdown(
        "Visualizing historical daily sales data to uncover trends and patterns.")

    # Prepare Day Names for Chart 2
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
                 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df['day_name'] = df['day_of_week'].map(day_names)
    df['day_name'] = pd.Categorical(
        df['day_name'], categories=list(day_names.values()), ordered=True)

    # Create a 2x2 grid for charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Distribution of Daily Revenue")
        fig1 = px.histogram(df, x='sales_revenue', nbins=40, marginal='box',
                            title='Daily Sales Revenue Distribution',
                            labels={'sales_revenue': 'Revenue ($)'})
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("2. Revenue by Day of Week")
        dow_sales = df.groupby('day_name', observed=False)[
            'sales_revenue'].mean().reset_index()
        fig2 = px.bar(dow_sales, x='day_name', y='sales_revenue',
                      title='Average Revenue by Day',
                      labels={'sales_revenue': 'Avg Revenue ($)'})
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("3. Revenue by Product Category")
        cat_sales = df.groupby('Category')['sales_revenue'].mean(
        ).reset_index().sort_values('sales_revenue', ascending=False)
        fig3 = px.bar(cat_sales, x='Category', y='sales_revenue',
                      title='Average Revenue by Category',
                      labels={'sales_revenue': 'Avg Revenue ($)'})
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("4. Feature Correlation Heatmap")
        numeric_cols = ['sales_revenue', 'previous_sales',
                        'num_transactions', 'promotion_rate', 'day_of_week']
        corr_matrix = df[numeric_cols].corr()
        fig4 = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                         title='Correlation Matrix',
                         color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig4, use_container_width=True)


# ==========================================
# 5. PAGE 2: AI PREDICTOR
# ==========================================
elif page == "🤖 AI Predictor":
    st.title("🤖 AI Monthly Revenue Simulator")
    st.markdown("""
    Enter the expected business parameters for the upcoming month. Our **Random Forest Regression Model** 
    will predict the total expected sales revenue for your chosen product category.
    """)

    st.info("💡 *Note: This model predicts total **Monthly Revenue**, not the individual unit price of a single item.*")

    # Input Columns
    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox(
            "📦 Product Category",
            ["Furniture", "Office Supplies", "Technology"]
        )
        num_transactions = st.number_input(
            "🛒 Expected Number of Transactions (Quantity)",
            min_value=0, value=100, step=10,
            help="Total number of items you expect to sell this month."
        )

    with col2:
        promotion_rate = st.slider(
            "🏷️ Expected Promotion Rate",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Percentage of items sold on discount (0.0 = no discounts, 1.0 = everything on sale)."
        )
        previous_month_sales = st.number_input(
            "💰 Previous Month's Sales Revenue ($)",
            min_value=0.0, value=5000.0, step=100.0,
            help="Total revenue generated in the immediate prior month."
        )

    # Predict Button
    if st.button("🚀 Predict Monthly Revenue", type="primary", use_container_width=True):

        # 1. Prepare the input data to match the exact format the model was trained on
        # The model expects: num_transactions, promotion_rate, previous_month_sales, Category_Office Supplies, Category_Technology
        input_data = {
            'num_transactions': [num_transactions],
            'promotion_rate': [promotion_rate],
            'previous_month_sales': [previous_month_sales],
            'Category_Office Supplies': [1 if category == "Office Supplies" else 0],
            'Category_Technology': [1 if category == "Technology" else 0]
        }

        input_df = pd.DataFrame(input_data)

        # 2. Scale the input using the saved scaler
        input_scaled = scaler.transform(input_df)

        # 3. Make the prediction
        prediction = model.predict(input_scaled)[0]

        # 4. Display the result
        st.markdown("---")
        st.success(
            f"### 🎉 Predicted Monthly Sales Revenue: ${prediction:,.2f}")
        st.markdown(
            f"**Category:** {category} | **Transactions:** {num_transactions} | **Promo Rate:** {promotion_rate*100:.0f}%")
