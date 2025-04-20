import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image
from datetime import datetime, timedelta
import io
import os
import base64

# Import utility modules
from utils import load_and_preprocess, format_currency, get_sales_status
from utils_report import create_pdf_report, get_download_link, export_to_excel, get_excel_download_link
from visualizations import create_mrp_plot, create_outlet_size_plot, create_outlet_type_plot, create_correlation_heatmap
from additional_visualizations import (create_feature_importance_plot, create_product_category_plot, 
                                   create_outlet_comparison_plot, create_mrp_sales_scatter)
from time_series import generate_time_series_data, create_time_series_plot, forecast_sales

# Import the advanced model
from models.advanced_model import model
dataset_path = "datasets/train.csv"

# Set page config
st.set_page_config(
    page_title="Retail Forecasting Dashboard",
    layout="wide",
    page_icon="🛍️",
    initial_sidebar_state="expanded"
)

# Custom styling - Enhanced for mobile
st.markdown("""
    <style>
        /* Base styling */
        .main {
            background-color: #f7f9fb;
            padding: 0.5rem;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 100%;
        }
        
        /* Typography */
        h1 {
            color: #2E3B55;
            font-size: calc(1.5rem + 0.5vw);
            margin-bottom: 1rem;
        }
        h2 {
            color: #3a506b;
            font-size: calc(1.2rem + 0.3vw);
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        h3 {
            color: #3a506b;
            font-size: calc(1rem + 0.2vw);
            font-weight: 500;
        }
        p {
            font-size: calc(0.9rem + 0.1vw);
            line-height: 1.5;
        }
        
        /* Card styling */
        .css-1r6slb0 {
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        /* Form elements */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 5px;
            font-weight: 500;
            width: 100%;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stSelectbox, .stNumberInput {
            margin-bottom: 0.75rem;
        }
        
        /* Mobile-specific adjustments */
        @media (max-width: 768px) {
            .main {
                padding: 0.25rem;
            }
            .block-container {
                padding: 0.5rem;
            }
            div[data-testid="stHorizontalBlock"] > div {
                min-width: 100%;
            }
            .css-1r6slb0 {
                padding: 0.75rem;
            }
            .stDataFrame {
                overflow-x: auto;
            }
        }
        
        /* Status messages */
        div.stAlert p {
            font-size: 1rem;
            padding: 10px;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: auto;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        
        /* Metric cards */
        [data-testid="stMetric"] {
            background-color: white;
            padding: 15px 10px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 0.5rem;
        }
        
        /* Table styling */
        .dataframe {
            font-size: 0.9rem;
        }
        
        /* Improve container spacing */
        div.row-widget.stRadio > div {
            flex-direction: row;
            align-items: center;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with navigation
with st.sidebar:
    st.markdown("""
        <h2 style="text-align: center; margin-bottom: 20px;">Retail Forecasting</h2>
    """, unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Predict Sales"],
        icons=["bar-chart-line", "graph-up-arrow"],
        menu_icon="shop",
        default_index=0,
        styles={
            "container": {"padding": "0px", "border-radius": "5px"},
            "icon": {"font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#4CAF50"},
        }
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-top: 30px;">
        <p style="margin-bottom: 5px; font-weight: bold;">Retail Forecasting Tool</p>
        <p style="font-size: 0.8rem;">Optimize your business decisions</p>
    </div>
    """, unsafe_allow_html=True)

# =================== Dashboard ===================
if selected == "Dashboard":
    # Add tabs for all visualizations
    dashboard_tabs = st.tabs([
        "📊 Overview", 
        "🔍 Advanced Insights", 
        "⏱️ Time Series Analysis", 
        "📝 Reports"
    ])
    
    # Load data
    df = load_and_preprocess(dataset_path)
    st.write('✅ DataFrame loaded. Shape:', df.shape)
    
    # -------- Overview Tab --------
    with dashboard_tabs[0]:
        st.header("📊 Retail Sales Overview")
        st.markdown("Analyze historical sales data and discover trends")
        
        # Key metrics
        st.markdown("### 📈 Key Performance Indicators")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric(
                label="Avg Sales Value",
                value=f"₹{df['Item_Outlet_Sales'].mean():.2f}",
                delta=f"{((df['Item_Outlet_Sales'].mean() / 2000) - 1) * 100:.1f}%"
            )
        
        with metrics_col2:
            st.metric(
                label="Total Items",
                value=f"{df['Item_Identifier'].nunique()}",
                delta=None
            )
        
        with metrics_col3:
            st.metric(
                label="Total Outlets",
                value=f"{df['Outlet_Identifier'].nunique()}",
                delta=None
            )
        
        with metrics_col4:
            st.metric(
                label="Total Records",
                value=f"{len(df):,}",
                delta=None
            )
        
        st.markdown("---")
        
        # Dataset preview with filter
        with st.expander("📋 Dataset Preview", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col2:
                n_rows = st.number_input("Number of rows", min_value=5, max_value=20, value=5)
            
            with col1:
                view_option = st.radio(
                    "View",
                    options=["Random Sample", "Head", "Tail"],
                    horizontal=True
                )
            
            if view_option == "Random Sample":
                displayed_data = df.sample(n_rows)
            elif view_option == "Head":
                displayed_data = df.head(n_rows)
            else:
                displayed_data = df.tail(n_rows)
            
            st.dataframe(
                displayed_data,
                use_container_width=True 
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Data Shape:", df.shape)
            with col2:
                st.write("Missing Values:", df.isna().sum().sum())
        
        # Summary Statistics
        with st.expander("📊 Summary Statistics", expanded=False):
            st.dataframe(
                df.describe().style.background_gradient(cmap="Blues"),
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Visualization section
        st.subheader("📊 Basic Sales Insights")
        
        # Tabs for basic visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "MRP Distribution", 
            "Sales by Outlet Size", 
            "Sales by Outlet Type",
            "Correlation Analysis"
        ])
        
        with tab1:
            fig = create_mrp_plot(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('⚠️ Could not generate plot: ' + str(create_mrp_plot(df)))
        
        with tab2:
            fig = create_outlet_size_plot(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('⚠️ Could not generate plot: ' + str(create_outlet_size_plot(df)))
        
        with tab3:
            fig = create_outlet_type_plot(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('⚠️ Could not generate plot: ' + str(create_outlet_type_plot(df)))
            
        with tab4:
            fig = create_correlation_heatmap(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('⚠️ Could not generate plot: ' + str(create_correlation_heatmap(df)))
    
    # -------- Advanced Insights Tab --------
    with dashboard_tabs[1]:
        st.header("🔍 Advanced Analytics Insights")
        st.markdown("Deeper analysis of sales patterns and relationships")
        
        # Feature Importance Analysis
        st.subheader("Feature Importance Analysis")
        st.markdown("Understanding key factors that influence sales performance")
        
        # Try to get feature importance from model
        try:
            feature_importance = model.get_feature_importance()
            if not feature_importance:
                # If empty, create some default feature importance
                feature_importance = {
                    "Item_MRP": 0.35,
                    "Outlet_Type": 0.25,
                    "Outlet_Size": 0.20,
                    "Outlet_Age": 0.15,
                    "Item_Visibility": 0.05
                }
            
            # Display feature importance plot
            st.plotly_chart(
                create_feature_importance_plot(feature_importance),
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"Could not retrieve feature importance: {e}")
            
        # Additional insights with two columns for compact mobile view
        col1, col2 = st.columns(2)
        
        with col1:
            # Product category analysis
            st.subheader("Product Category Analysis")
            st.plotly_chart(
                create_product_category_plot(df),
                use_container_width=True
            )
        
        with col2:
            # Price vs Sales relationship
            st.subheader("Price vs Sales Relationship")
            st.plotly_chart(
                create_mrp_sales_scatter(df),
                use_container_width=True
            )
        
        # Outlet comparison (full width)
        st.subheader("Outlet Performance Comparison")
        st.markdown("Radar chart comparing different outlets across multiple metrics")
        st.plotly_chart(
            create_outlet_comparison_plot(df),
            use_container_width=True
        )
    
    # -------- Time Series Tab --------
    with dashboard_tabs[2]:
        st.header("⏱️ Time Series Analysis")
        st.markdown("Analyze sales trends over time and forecast future performance")
        
        # Generate time series data
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Options for time series
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now() - timedelta(days=30)
            )
            
            days_range = st.slider(
                "Date Range (days)",
                min_value=90,
                max_value=730,
                value=365,
                step=30
            )
            
            forecast_weeks = st.slider(
                "Forecast Horizon (weeks)",
                min_value=4,
                max_value=24,
                value=12,
                step=4
            )
        
        with col1:
            # Generate time series data based on parameters
            ts_data = generate_time_series_data(
                df, 
                start_date=start_date.strftime('%Y-%m-%d'),
                days=days_range
            )
            
            # Display time series plot
            st.subheader("Historical Sales Trend")
            st.plotly_chart(
                create_time_series_plot(ts_data),
                use_container_width=True
            )
        
        # Add forecasting section
        st.subheader("Sales Forecast")
        st.markdown("Projection of future sales based on historical patterns")
        
        # Generate forecast
        forecast_fig, forecast_data = forecast_sales(ts_data, periods=forecast_weeks)
    if forecast_fig:
        st.plotly_chart(forecast_fig, use_container_width=True)
    else:
        st.warning('⚠️ Forecast plot could not be generated')
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Show forecast data in a table
        with st.expander("View Forecast Data", expanded=False):
            st.dataframe(
                forecast_data.style.format({
                    'Forecasted_Sales': '₹{:.2f}',
                    'Lower_Bound': '₹{:.2f}',
                    'Upper_Bound': '₹{:.2f}'
                }),
                use_container_width=True
            )
    
    # -------- Reports Tab --------
    with dashboard_tabs[3]:
        st.header("📝 Reports & Exports")
        st.markdown("Generate and download customized reports")
        
        # Options for report generation
        st.subheader("Configure Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_format = st.radio(
                "Report Format",
                options=["PDF Report", "Excel Data Export"],
                horizontal=True
            )
            
        with col2:
            include_predictions = st.checkbox("Include Sales Predictions", value=True)
            include_feature_importance = st.checkbox("Include Feature Importance", value=True)
        
        # Sample prediction results for report
        if include_predictions:
            # Create sample prediction for report
            sample_features = np.array([[150.0, 3, 1, 2, 15]])  # Sample input values
            sample_prediction = float(model.predict(sample_features)[0])
            
            prediction_info = {
                'inputs': {
                    'Item MRP': '₹150.00',
                    'Outlet': 'OUT018',
                    'Outlet Size': 'Medium',
                    'Outlet Type': 'Supermarket Type2',
                    'Outlet Age': '15 years'
                },
                'prediction': sample_prediction,
                'lower_bound': sample_prediction * 0.85,
                'upper_bound': sample_prediction * 1.15
            }
        else:
            prediction_info = None
        
        # Feature importance for report
        if include_feature_importance:
            try:
                feature_imp = model.get_feature_importance()
                if not feature_imp:
                    feature_imp = {
                        "Item_MRP": 0.35,
                        "Outlet_Type": 0.25,
                        "Outlet_Size": 0.20,
                        "Outlet_Age": 0.15,
                        "Item_Visibility": 0.05
                    }
            except:
                feature_imp = {
                    "Item_MRP": 0.35,
                    "Outlet_Type": 0.25,
                    "Outlet_Size": 0.20,
                    "Outlet_Age": 0.15,
                    "Item_Visibility": 0.05
                }
        else:
            feature_imp = None
        
        # Generate report button
        if st.button("Generate Report", type="primary"):
            # Display spinner while generating
            with st.spinner("Generating report..."):
                if report_format == "PDF Report":
                    # Generate PDF report
                    pdf_bytes = create_pdf_report(df, prediction_info, feature_imp)
                    
                    # Display download link
                    st.markdown(
                        get_download_link(pdf_bytes, "retail_forecast_report.pdf"),
                        unsafe_allow_html=True
                    )
                    
                    st.success("PDF report generated successfully!")
                else:
                    # Generate Excel export
                    excel_bytes = export_to_excel(df, prediction_info)
                    
                    # Display download link
                    st.markdown(
                        get_excel_download_link(excel_bytes, "retail_forecast_data.xlsx"),
                        unsafe_allow_html=True
                    )
                    
                    st.success("Excel data export generated successfully!")

# =================== Predict Sales ===================
elif selected == "Predict Sales":
    st.title("🔮 Sales Prediction Tool")
    st.markdown("Enter item and outlet attributes to forecast sales value")
    
    # Create tabs for different prediction options
    predict_tabs = st.tabs(["Basic Prediction", "Advanced Options"])
    
    with predict_tabs[0]:
        # Form in a card-like container
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
            <h3 style='margin-bottom: 15px;'>Enter Product & Outlet Details</h3>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("predict_form"):
            # Split form into columns for better mobile layout
            col1, col2 = st.columns(2)
            
            with col1:
                item_mrp = st.number_input(
                    "Item MRP (₹)", 
                    min_value=30.0, 
                    max_value=270.0,
                    value=140.0,
                    step=10.0,
                    help="Retail price of the product"
                )
                
                outlet_identifier = st.selectbox(
                    "Outlet Identifier", 
                    ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 
                     'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'],
                    help="Unique ID of the outlet"
                )
                
                outlet_size = st.selectbox(
                    "Outlet Size", 
                    ['Small', 'Medium', 'High'],
                    help="Size classification of the outlet"
                )
            
            with col2:
                outlet_type = st.selectbox(
                    "Outlet Type", 
                    ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'],
                    help="Type of outlet"
                )
                
                est_year = st.number_input(
                    "Outlet Establishment Year", 
                    min_value=1985, 
                    max_value=2024, 
                    value=2000,
                    step=1,
                    help="Year the outlet was established"
                )
            
            # Full-width submit button
            submitted = st.form_submit_button("🔍 Predict Sales")
        
        # Display prediction results with visualization
        if submitted:
            # Convert categorical values to numerical
            outlet_id_map = {k: i for i, k in enumerate(['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'])}
            outlet_size_map = {'High': 0, 'Medium': 1, 'Small': 2}
            outlet_type_map = {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}
            
            outlet_id = outlet_id_map[outlet_identifier]
            outlet_sz = outlet_size_map[outlet_size]
            outlet_tp = outlet_type_map[outlet_type]
            outlet_age = 2025 - est_year
            
            # Prepare features and predict
            features = np.array([[item_mrp, outlet_id, outlet_sz, outlet_tp, outlet_age]])
            prediction = model.predict(features)[0]
            
            # Format the prediction
            prediction_val = float(prediction)
            lower_bound = prediction_val * 0.85  # 15% lower bound
            upper_bound = prediction_val * 1.15  # 15% upper bound
            
            # Display prediction in a nice card with gradient based on value
            status = get_sales_status(prediction_val)
            status_color = {
                'low': '#e74c3c',    # Red
                'medium': '#f39c12', # Orange
                'high': '#2ecc71'    # Green
            }
            
            # Create two columns for prediction result and explanation
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown(f"""
                    <div style='background-color: white; padding: 20px; border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-top: 20px; height: 100%;'>
                        <h3 style='margin-bottom: 10px;'>Prediction Results</h3>
                        <p>Based on the provided attributes, the predicted sales value is:</p>
                        <div style='background-color: {status_color[status]}; color: white; padding: 15px; 
                        border-radius: 5px; text-align: center; font-size: 24px; font-weight: bold;'>
                            ₹{prediction_val:.2f}
                        </div>
                        <p style='margin-top: 15px;'>
                            <b>Expected Range:</b> ₹{lower_bound:.2f} to ₹{upper_bound:.2f}
                        </p>
                        <p style='margin-top: 15px;'>
                            <b>Status:</b> <span style='color: {status_color[status]}; font-weight: bold;'>{status.title()}</span>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div style='background-color: white; padding: 20px; border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-top: 20px; height: 100%;'>
                        <h3 style='margin-bottom: 10px;'>Input Summary</h3>
                        <p>
                            <b>Item Price:</b> ₹{item_mrp:.2f}<br><br>
                            <b>Outlet:</b> {outlet_identifier}<br>
                            <b>Outlet Type:</b> {outlet_type}<br>
                            <b>Size:</b> {outlet_size}<br>
                            <b>Established:</b> {est_year}<br>
                            <b>Outlet Age:</b> {outlet_age} years
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Create a PDF report for this prediction
            if st.button("📄 Generate Prediction Report"):
                with st.spinner("Generating report..."):
                    # Prepare prediction info
                    prediction_info = {
                        'inputs': {
                            'Item MRP': f'₹{item_mrp:.2f}',
                            'Outlet': outlet_identifier,
                            'Outlet Size': outlet_size,
                            'Outlet Type': outlet_type,
                            'Outlet Age': f'{outlet_age} years'
                        },
                        'prediction': prediction_val,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                    
                    # Get feature importance if available
                    try:
                        feature_imp = model.get_feature_importance()
                    except:
                        feature_imp = {
                            "Item_MRP": 0.35,
                            "Outlet_Type": 0.25,
                            "Outlet_Size": 0.20,
                            "Outlet_Age": 0.15,
                            "Item_Visibility": 0.05
                        }
                    
                    # Load data for report
                    df = load_and_preprocess(dataset_path)
    st.write('✅ DataFrame loaded. Shape:', df.shape)
                    
                    # Generate PDF report
                    pdf_bytes = create_pdf_report(df, prediction_info, feature_imp)
                    
                    # Display download link
                    st.markdown(
                        get_download_link(pdf_bytes, "sales_prediction_report.pdf"),
                        unsafe_allow_html=True
                    )
                    
                    st.success("PDF report generated successfully!")
            
            # Add a disclaimer
            st.info(
                "This prediction is based on historical data patterns. Actual sales may vary due to additional factors not considered in this model."
            )
    
    with predict_tabs[1]:
        st.subheader("Advanced Prediction Options")
        
        # Batch prediction section
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
            <h3 style='margin-bottom: 15px;'>Batch Prediction</h3>
            <p>Upload a CSV file with multiple records to get predictions in batch.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # File uploader for batch prediction
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load data
                batch_data = pd.read_csv(uploaded_file)
                
                # Show preview
                st.write("Preview of uploaded data:")
                st.dataframe(batch_data.head(), use_container_width=True)
                
                # Display required columns info
                st.info(
                    "Your CSV should have columns: 'Item_MRP', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Type', 'Outlet_Establishment_Year'"
                )
                
                # Check if process button is clicked
                if st.button("Process Batch Predictions"):
                    # Check if required columns exist
                    required_columns = ['Item_MRP', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Type', 'Outlet_Establishment_Year']
                    missing_columns = [col for col in required_columns if col not in batch_data.columns]
                    
                    if missing_columns:
                        st.error(f"Error: Missing required columns: {', '.join(missing_columns)}")
                    else:
                        # Process batch
                        with st.spinner("Processing batch predictions..."):
                            # Mapping dictionaries
                            outlet_id_map = {k: i for i, k in enumerate(['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'])}
                            outlet_size_map = {'High': 0, 'Medium': 1, 'Small': 2}
                            outlet_type_map = {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}
                            
                            # Process each row
                            predictions = []
                            for _, row in batch_data.iterrows():
                                try:
                                    # Extract features
                                    item_mrp = float(row['Item_MRP'])
                                    outlet_id = outlet_id_map.get(row['Outlet_Identifier'], 0)
                                    outlet_sz = outlet_size_map.get(row['Outlet_Size'], 1)
                                    outlet_tp = outlet_type_map.get(row['Outlet_Type'], 1)
                                    
                                    # Calculate outlet age
                                    try:
                                        est_year = int(row['Outlet_Establishment_Year'])
                                        outlet_age = 2025 - est_year
                                    except:
                                        outlet_age = 10  # Default
                                    
                                    # Predict
                                    features = np.array([[item_mrp, outlet_id, outlet_sz, outlet_tp, outlet_age]])
                                    prediction = float(model.predict(features)[0])
                                    predictions.append(prediction)
                                except Exception as e:
                                    predictions.append(None)
                            
                            # Add predictions to dataframe
                            batch_data['Predicted_Sales'] = predictions
                            
                            # Show results
                            st.subheader("Batch Prediction Results")
                            st.dataframe(
                                batch_data.style.format({'Predicted_Sales': '₹{:.2f}'}),
                                use_container_width=True
                            )
                            
                            # Create download link for results
                            csv = batch_data.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="batch_predictions.csv">Download results as CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
        
        # Model information
        st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-top: 20px;'>
            <h3 style='margin-bottom: 15px;'>Model Information</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Get feature importance
        try:
            feature_importance = model.get_feature_importance()
            if feature_importance:
                st.subheader("Feature Importance")
                st.plotly_chart(
                    create_feature_importance_plot(feature_importance),
                    use_container_width=True
                )
            else:
                st.write("Feature importance information not available.")
        except:
            st.write("Feature importance information not available for this model.")
