import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- CONFIGURATION ---
st.set_page_config(page_title="House Price AI", page_icon="🏠", layout="wide")

# --- 1. LOAD DATA (Now with Outlier Removal) ---
@st.cache_data
def load_data():
    try:
        # Load the CSV
        df = pd.read_csv("train.csv")
        
        # 1. CLEANING: Remove massive outliers that confuse the model
        # (Houses > 4000 sq ft that sold for cheap)
        df = df[df['GrLivArea'] < 4000]
        
        # 2. SELECTION: We added 'FullBath' and 'YearRemodAdd' for better accuracy
        features = ['OverallQual', 'GrLivArea', 'GarageCars', 
                    'TotalBsmtSF', 'YearBuilt', 'FullBath', 'YearRemodAdd', 'SalePrice']
        
        df = df[features].dropna()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- 2. TRAIN MODEL (Switched to Random Forest) ---
def train_model(df):
    X = df[['OverallQual', 'GrLivArea', 'GarageCars', 
            'TotalBsmtSF', 'YearBuilt', 'FullBath', 'YearRemodAdd']]
    y = df['SalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # UPGRADE: RandomForest is smarter than Linear Regression
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    accuracy = r2_score(y_test, model.predict(X_test))
    
    return model, accuracy

# --- 3. INTERFACE ---
st.title("🏠 House Price Predictor (Pro)")
st.markdown("Enter the details of the house below to estimate its price using a **Random Forest AI Model**.")

df = load_data()

if df is not None:
    model, accuracy = train_model(df)
    
    # --- INPUTS (Organized in Columns) ---
    c1, c2, c3 = st.columns(3)
    
    with c1:
        quality = st.slider("Overall Quality (1-10)", 1, 10, 5)
        year = st.number_input("Year Built", 1900, 2024, 2000)
        remod = st.number_input("Year Remodeled", 1900, 2024, 2005) # NEW INPUT
        
    with c2:
        area = st.number_input("Living Area (sq ft)", 500, 4000, 1500)
        basement = st.number_input("Basement Area (sq ft)", 0, 3000, 1000)
        
    with c3:
        garage = st.selectbox("Garage Capacity (Cars)", [0, 1, 2, 3, 4])
        baths = st.selectbox("Full Bathrooms", [1, 2, 3, 4]) # NEW INPUT

    st.divider()

    # --- PREDICTION ---
    if st.button("💰 Estimate Price", type="primary"):
        # Create input data matching the training columns EXACTLY
        input_data = pd.DataFrame([[quality, area, garage, basement, year, baths, remod]], 
                                  columns=['OverallQual', 'GrLivArea', 'GarageCars', 
                                           'TotalBsmtSF', 'YearBuilt', 'FullBath', 'YearRemodAdd'])
        
        prediction = model.predict(input_data)[0]
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.success(f"Estimated Price: ${prediction:,.2f}")
        with col_res2:
            st.info(f"Model Accuracy: {accuracy:.1%}")
            
    # Show user that we are using better data now
    with st.expander("View Training Data Snapshot"):
        st.dataframe(df.head(10))
        
else:
    st.error("Could not find 'train.csv'. Please upload it to the repository.")
