import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Coal Stocks Analysis",
    page_icon="⚡",
    layout="wide"
)

# Load the data
@st.cache_data
def load_data():
    # Read the CSV file
    df = pd.read_csv('cleaned_daily-coal-stocks (8).csv')
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Convert numeric columns to appropriate types
    numeric_columns = ['capacity', 'daily_requirement', 'daily_receipt', 'daily_consumption',
                      'req_normative_stock', 'normative_stock_days', 'indigenous_stock',
                      'import_stock', 'total_stock', 'stock_days', 'plf_prcnt',
                      'actual_vs_normative_stock_prcnt']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert categorical columns to string type
    categorical_columns = ['state_name', 'state_code', 'power_station_name',
                         'sector', 'utility', 'mode_of_transport', 'remarks']
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df

# Load the data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Create sidebar for navigation
page = st.sidebar.selectbox(
    "Choose a page",
    ["About Dataset", "Exploratory Data Analysis", "Model Prediction"]
)

# About Dataset page
if page == "About Dataset":
    st.title("Coal Stocks Dataset Analysis")
    
    st.markdown("""
    This application analyzes daily coal-stock information for various power stations.
    The dataset includes information about capacity, daily requirements, stock levels,
    and other operational metrics.
    """)
    
    # Show raw data preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # Display column information
    st.subheader("Column Information")
    
    # Create a buffer to capture the output of df.info()
    from io import StringIO
    buffer = StringIO()
    df.info(buf=buffer)
    
    # Display the captured output in a text area
    st.text(buffer.getvalue())
    
    # Display missing values summary
    st.subheader("Missing Values Summary")
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': df.isnull().sum(),
        'Percentage Missing': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(missing_data)
    
    # Display basic statistics
    st.subheader("Basic Statistics")
    st.dataframe(df.describe())

# EDA page
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    
    # Create tabs for different types of visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Categorical Analysis", "Correlations", "Scatter Plots"])
    
    with tab1:
        st.subheader("Distribution of Numerical Variables")
        
        # Distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Capacity distribution
        sns.histplot(data=df, x='capacity', ax=axes[0,0])
        axes[0,0].set_title('Distribution of Capacity')
        
        # Daily requirement distribution
        sns.histplot(data=df, x='daily_requirement', ax=axes[0,1])
        axes[0,1].set_title('Distribution of Daily Requirement')
        
        # Stock days distribution
        sns.histplot(data=df, x='stock_days', ax=axes[1,0])
        axes[1,0].set_title('Distribution of Stock Days')
        
        # PLF percent distribution
        sns.histplot(data=df, x='plf_prcnt', ax=axes[1,1])
        axes[1,1].set_title('Distribution of PLF Percent')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Categorical Variable Analysis")
        
        # Create plots for categorical variables
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sector distribution
        df['sector'].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_title('Distribution by Sector')
        ax1.set_xlabel('Sector')
        ax1.set_ylabel('Count')
        
        # Mode of transport distribution
        df['mode_of_transport'].value_counts().plot(kind='bar', ax=ax2)
        ax2.set_title('Distribution by Mode of Transport')
        ax2.set_xlabel('Mode of Transport')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Correlation Analysis")
        
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title('Correlation Heatmap')
        st.pyplot(fig)
    
    with tab4:
        st.subheader("Scatter Plots")
        
        # Capacity vs Daily Requirement
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='capacity', y='daily_requirement')
        plt.title('Capacity vs Daily Requirement')
        st.pyplot(fig)

# Model Prediction page
elif page == "Model Prediction":
    st.title("Random Forest Model Prediction")
    st.markdown("""
    Use the sliders below to input values and get a prediction for the daily coal requirement.
    This model was trained on the dataset and achieved the highest performance among all tested models.
    """)
    
    # Create input features
    col1, col2 = st.columns(2)
    
    with col1:
        capacity = st.slider("Capacity (MW)", 
                           float(df['capacity'].min()), 
                           float(df['capacity'].max()),
                           float(df['capacity'].mean()))
        
        daily_consumption = st.slider("Daily Consumption",
                                    float(df['daily_consumption'].min()),
                                    float(df['daily_consumption'].max()),
                                    float(df['daily_consumption'].mean()))
        
        total_stock = st.slider("Total Stock",
                              float(df['total_stock'].min()),
                              float(df['total_stock'].max()),
                              float(df['total_stock'].mean()))
    
    with col2:
        stock_days = st.slider("Stock Days",
                             float(df['stock_days'].min()),
                             float(df['stock_days'].max()),
                             float(df['stock_days'].mean()))
        
        plf_prcnt = st.slider("PLF Percent",
                             float(df['plf_prcnt'].min()),
                             float(df['plf_prcnt'].max()),
                             float(df['plf_prcnt'].mean()))
    
    # Prepare features for prediction
    features = ['capacity', 'daily_consumption', 'total_stock', 'stock_days', 'plf_prcnt']
    
    # Handle missing values
    X = df[features].copy()
    y = df['daily_requirement'].copy()
    
    # Remove rows with NaN values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    # Train the model
    @st.cache_resource
    def train_model(X, y):
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        rf_model.fit(X_scaled, y)
        return rf_model, scaler
    
    model, scaler = train_model(X, y)
    
    # Make prediction
    if st.button("Predict Daily Requirement"):
        input_data = np.array([[capacity, daily_consumption, total_stock, stock_days, plf_prcnt]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        st.success(f"Predicted Daily Requirement: {prediction:.2f} tonnes")
        
        # Feature importance
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title('Feature Importance in Prediction')
        st.pyplot(fig)