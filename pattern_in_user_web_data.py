import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from urllib.request import urlretrieve

# Download the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
urlretrieve(url, "online_shoppers_intention.csv")

# Load dataset
data = pd.read_csv("online_shoppers_intention.csv")

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

# Define function to make predictions
def predict_user_input(model, scaler, label_encoder, user_input):
    # Scale user input
    user_scaled = scaler.transform(user_input)

    # Make prediction
    prediction = model.predict(user_scaled)
    probability = model.predict_proba(user_scaled)[:, 1]

    # Decode prediction
    prediction_label = label_encoder.inverse_transform(prediction)[0]
    return prediction_label, probability[0]

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

# Weekend selection
user_input['Weekend'] = st.radio("Weekend", ["Yes", "No"])

# Convert 'Month' to numerical value
user_input['Month'] = months.index(user_input['Month']) + 1

# Convert 'Weekend' to binary
user_input['Weekend'] = 1 if user_input['Weekend'] == "Yes" else 0

# Drop unnecessary columns
user_input.pop('VisitorType_Returning_Visitor')

# Validate user input
if 'Month' not in user_input or 'Weekend' not in user_input:
    st.error("Please provide month and select weekend.")

# Button to make prediction
if st.button("Predict"):
    if 'Month' in user_input and 'Weekend' in user_input:
        # Convert user input to DataFrame
        user_df = pd.DataFrame([user_input])

        # Make prediction
        prediction_label, probability = predict_user_input(model, scaler, label_encoder, user_df)

        # Display prediction
        st.header("Prediction")
        st.write(f"Predicted Revenue: {prediction_label}")
        st.write(f"Probability of Revenue: {probability:.2f}")
