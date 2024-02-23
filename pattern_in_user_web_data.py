import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from urllib.request import urlretrieve

# Download the dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
# urlretrieve(url, "online_shoppers_intention.csv")

# Load dataset
data = pd.read_csv("shopper's detail.csv")

# Drop rows with missing values
data.dropna(inplace=True)

# Encode categorical variables
cat_cols = ['Month', 'VisitorType', 'Weekend']
data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# Encode target variable
label_encoder = LabelEncoder()
data_encoded['Revenue'] = label_encoder.fit_transform(data_encoded['Revenue'])

# Split data into features and target variable
X = data_encoded.drop('Revenue', axis=1)
y = data_encoded['Revenue']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Streamlit UI
st.title("Online Shopper's Purchase Intention Prediction")

# User input
st.header("Enter Session Information")

# Collect user input
user_input = {}
user_input['Administrative'] = st.number_input("Administrative", value=0)
user_input['Administrative_Duration'] = st.number_input("Administrative Duration", value=0)
user_input['Informational'] = st.number_input("Informational", value=0)
user_input['Informational_Duration'] = st.number_input("Informational Duration", value=0)
user_input['ProductRelated'] = st.number_input("Product Related", value=0)
user_input['ProductRelated_Duration'] = st.number_input("Product Related Duration", value=0)
user_input['BounceRates'] = st.number_input("Bounce Rates", value=0)
user_input['ExitRates'] = st.number_input("Exit Rates", value=0)
user_input['PageValues'] = st.number_input("Page Values", value=0)
user_input['SpecialDay'] = st.number_input("Special Day", value=0)

# Month dropdown
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
user_input['Month'] = st.selectbox("Month", months)

# Convert 'Month' to numerical value
user_input['Month'] = months.index(user_input['Month']) + 1

# Weekend selection
user_input['Weekend'] = st.radio("Weekend", ["Yes", "No"])

# Convert 'Weekend' to binary
user_input['Weekend'] = 1 if user_input['Weekend'] == "Yes" else 0

# Visitor Type selection
visitor_type = st.radio("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"])
user_input['VisitorType_Returning_Visitor'] = 1 if visitor_type == "Returning_Visitor" else 0

# Button to make prediction
if st.button("Predict"):
    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_input])

    # One-hot encode categorical variables
    user_df = pd.get_dummies(user_df, columns=['Month'], drop_first=True)

    # Ensure that all columns present during training are present in user input
    missing_cols = set(X.columns) - set(user_df.columns)
    for col in missing_cols:
        user_df[col] = 0

    # Reorder columns to match training data
    user_df = user_df[X.columns]

    # Scale user input
    user_scaled = scaler.transform(user_df)

    # Make prediction
    prediction = model.predict(user_scaled)[0]
    probability = model.predict_proba(user_scaled)[0][1]

    # Decode prediction
    prediction_label = label_encoder.inverse_transform([prediction])[0]

    # Display prediction
    st.header("Prediction")
    st.write(f"Predicted Revenue(trillion $ ;-)): {prediction_label}")
    st.write(f"Probability of Revenue: {probability:.2f}")
