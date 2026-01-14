import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# CONFIGURATION
st.set_page_config(page_title="House Price AI", page_icon="🏠", layout="wide")

# 1. LOAD DATA
@st.cache_data
def load_data():
    try:
        # Load the CSV you uploaded to GitHub
        df = pd.read_csv("train.csv")
        # Keep only the features we want to use
        features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt', 'SalePrice']
        df = df[features].dropna()
        return df
    except Exception as e:
        return None

# 2. TRAIN MODEL
def train_model(df):
    X = df[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']]
    y = df['SalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, r2_score(y_test, model.predict(X_test))

# 3. INTERFACE
st.title("🏠 House Price Predictor")
st.markdown("Enter the details of the house below to estimate its price using AI.")

df = load_data()

if df is not None:
    model, accuracy = train_model(df)
    
    # INPUTS
    c1, c2, c3 = st.columns(3)
    with c1:
        quality = st.slider("Overall Quality (1-10)", 1, 10, 5)
        year = st.number_input("Year Built", 1900, 2024, 2000)
    with c2:
        area = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
        basement = st.number_input("Basement (sq ft)", 0, 3000, 1000)
    with c3:
        garage = st.selectbox("Garage Cars", [0, 1, 2, 3, 4])

    # PREDICT BUTTON
    if st.button("💰 Estimate Price", type="primary"):
        input_data = pd.DataFrame([[quality, area, garage, basement, year]], 
                                  columns=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt'])
        prediction = model.predict(input_data)[0]
        
        st.success(f"Estimated Price: ${prediction:,.2f}")
        st.caption(f"Model Accuracy: {accuracy:.0%}")
        
else:
    st.error("Could not find 'train.csv'. Please upload it to the repository.")
