import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv("D:/project/retail_sales_data_2023_2025.csv")

# Preprocess Date column for time-series analysis
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date')
df = df.set_index('Date')

# Group data by date and calculate daily total sales
sales_data = df['Total Sales'].resample('D').sum()

# Streamlit App: Retail Sales Dashboard
st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")

# Advanced CSS for styling
st.markdown("""
    <style>
        /* Global Background */
        body {
            background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
            font-family: 'Arial', sans-serif;
            color: #333;
        }
        
        /* Header Title */
        .title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
            margin-top: 20px;
        }
        
        /* Metrics styling */
        .stMetricValue {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        
        /* Card Style */
        .reportview-container .main .block-container {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        }

        /* Sidebar Styling */
        .sidebar .sidebar-content {
            background-color: #f7f9fc;
            padding: 20px;
            border-radius: 12px;
        }
        
        /* Visualization Titles */
        .stMarkdown h2 {
            color: #333;
            font-size: 24px;
            border-left: 5px solid #4CAF50;
            padding-left: 10px;
        }
        
        /* Footer */
        footer {
            font-size: 12px;
            text-align: center;
            margin-top: 20px;
            color: #777;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="title">📊 Retail Sales Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Forecasting Parameters")
forecast_period = st.sidebar.slider("Forecast Period (days)", min_value=1, max_value=365, value=30)

# Visualization 1: Sales Over Time (Line Plot)
st.subheader("1. Sales Over Time")
fig = px.line(sales_data.reset_index(), x='Date', y='Total Sales', title="Total Sales Over Time",
              labels={'Date': 'Date', 'Total Sales': 'Sales ($)'})
st.plotly_chart(fig)

# Visualization 2: Correlation Heatmap
st.subheader("2. Correlation Heatmap")
plt.figure(figsize=(10, 6))
heatmap_data = df[['Price', 'Quantity Sold', 'Discount (%)', 'Loyalty Score', 'Total Sales']].corr()
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
st.pyplot(plt)

# Visualization 3: Sales by Region (Pie Chart)
st.subheader("3. Sales by Region")
region_sales = df.groupby('Region')['Total Sales'].sum().reset_index()
region_fig = px.pie(region_sales, names='Region', values='Total Sales', title="Sales Distribution by Region")
region_fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(region_fig)

# Visualization 4: Sales by Category (Bar Chart)
st.subheader("4. Sales by Category")
category_sales = df.groupby('Category')['Total Sales'].sum().reset_index()
category_fig = px.bar(category_sales, x='Category', y='Total Sales', title="Total Sales by Category",
                      labels={'Total Sales': 'Sales ($)', 'Category': 'Category'}, color='Category')
st.plotly_chart(category_fig)

# Visualization 5: Daily Sales Trend (Rolling Average)
st.subheader("5. Daily Sales Trend (7-Day Rolling Average)")
sales_data_rolling = sales_data.rolling(7).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=sales_data.index, y=sales_data, mode='lines', name='Daily Sales', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=sales_data_rolling.index, y=sales_data_rolling, mode='lines', name='7-Day Rolling Avg', line=dict(color='orange')))
fig.update_layout(title="Daily Sales with Rolling Average", xaxis_title="Date", yaxis_title="Sales ($)")
st.plotly_chart(fig)

# Visualization 6: Top 10 Products by Sales
st.subheader("6. Top 10 Products by Sales")
top_products = df.groupby('Product Name')['Total Sales'].sum().sort_values(ascending=False).head(10).reset_index()
fig = px.bar(top_products, x='Total Sales', y='Product Name', orientation='h', title="Top 10 Products by Sales",
             labels={'Total Sales': 'Sales ($)', 'Product Name': 'Product'}, color='Total Sales')
st.plotly_chart(fig)

# Visualization 7: Time-Series Decomposition
st.subheader("7. Time-Series Decomposition (Trend, Seasonality, Residuals)")
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(sales_data.dropna(), model='additive', period=7)
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
decomposition.trend.plot(ax=axes[0], title="Trend")
decomposition.seasonal.plot(ax=axes[1], title="Seasonality")
decomposition.resid.plot(ax=axes[2], title="Residuals")
plt.tight_layout()
st.pyplot(fig)

# Visualization 8: ARIMA Forecasting
st.subheader("8. ARIMA Forecast")
arima_model = ARIMA(sales_data, order=(1, 1, 1))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=forecast_period)

# Plot ARIMA Forecast
fig = go.Figure()
fig.add_trace(go.Scatter(x=sales_data.index, y=sales_data, mode='lines', name='Actual Sales', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=pd.date_range(sales_data.index[-1], periods=forecast_period, freq='D'),
                         y=arima_forecast, mode='lines', name='ARIMA Forecast', line=dict(color='orange')))
fig.update_layout(title="ARIMA Forecast", xaxis_title="Date", yaxis_title="Total Sales ($)")
st.plotly_chart(fig)

# Visualization 9: SARIMA Forecasting
st.subheader("9. SARIMA Forecast")
sarima_model = SARIMAX(sales_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
sarima_result = sarima_model.fit()
sarima_forecast = sarima_result.forecast(steps=forecast_period)

# Plot SARIMA Forecast
fig = go.Figure()
fig.add_trace(go.Scatter(x=sales_data.index, y=sales_data, mode='lines', name='Actual Sales', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=pd.date_range(sales_data.index[-1], periods=forecast_period, freq='D'),
                         y=sarima_forecast, mode='lines', name='SARIMA Forecast', line=dict(color='green')))
fig.update_layout(title="SARIMA Forecast", xaxis_title="Date", yaxis_title="Total Sales ($)")
st.plotly_chart(fig)

# Visualization 10: Sales Distribution (Histogram)
st.subheader("10. Sales Distribution")
dist_fig = px.histogram(df, x='Total Sales', nbins=50, title="Sales Distribution",
                        labels={'Total Sales': 'Sales ($)'}, marginal="box", color_discrete_sequence=['orange'])
st.plotly_chart(dist_fig)
