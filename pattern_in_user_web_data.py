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

def print_user_guide():
    ("Welcome! Please provide the following information for predicting online shopper's purchase intention:")
    print("\n1. Administrative: Number of pages related to account management or settings visited during your session.")
    print("   Example: Login page, account settings page.")
    print("2. Administrative Duration: Total time (in seconds) spent on administrative pages.")
    print("3. Informational: Number of pages containing general information visited during your session.")
    print("   Example: FAQ page, help guide.")
    print("4. Informational Duration: Total time (in seconds) spent on informational pages.")
    print("5. ProductRelated: Number of pages related to products or services visited during your session.")
    print("   Example: Product listing page, product detail page.")
    print("6. ProductRelated Duration: Total time (in seconds) spent on product-related pages.")
    print("7. BounceRates: Percentage of visitors who leave the website after viewing only one page.")
    print("   A high bounce rate may indicate visitors did not find the content engaging.")
    print("8. ExitRates: Percentage of visitors who leave the website from a specific page.")
    print("   It helps identify pages where users are most likely to exit the site.")
    print("9. PageValues: Average value of the pages visited during your session.")
    print("   Higher page values indicate pages contributing more to conversions.")
    print("10. SpecialDay: Closeness of the session to a special day (normalized).")
    print("    Example: Valentine's Day, Black Friday.")
    print("11. Month_Dec to Month_Sep: Presence (1) or absence (0) of each month in your session.")
    print("12. OperatingSystems: Code representing the operating system you used during your session.")
    print("13. Browser: Code representing the web browser you used during your session.")
    print("14. Region: Code representing your geographical region.")
    print("15. TrafficType: Code representing the type of traffic source through which you arrived at the website.")
    print("16. VisitorType_Other: Whether you are a returning visitor or fall into another visitor type category.")
    print("17. VisitorType_Returning_Visitor: Whether you are a returning visitor or not.")
    print("18. Weekend_True: Whether your session occurred on a weekend (1) or not (0).")


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
user_input['Administrative_Duration'] = st.number_input("Administrative Duration", value=0.0)
user_input['Informational'] = st.number_input("Informational", value=0)
user_input['Informational_Duration'] = st.number_input("Informational Duration", value=0.0)
user_input['ProductRelated'] = st.number_input("Product Related", value=0)
user_input['ProductRelated_Duration'] = st.number_input("Product Related Duration", value=0.0)
user_input['BounceRates'] = st.number_input("Bounce Rates", value=0.0)
user_input['ExitRates'] = st.number_input("Exit Rates", value=0.0)
user_input['PageValues'] = st.number_input("Page Values", value=0.0)
user_input['SpecialDay'] = st.number_input("Special Day", value=0.0)
user_input['Month_Dec'] = st.radio("Month_Dec", [0, 1])
user_input['Month_Feb'] = st.radio("Month_Feb", [0, 1])
user_input['Month_Jul'] = st.radio("Month_Jul", [0, 1])
user_input['Month_June'] = st.radio("Month_June", [0, 1])
user_input['Month_Mar'] = st.radio("Month_Mar", [0, 1])
user_input['Month_May'] = st.radio("Month_May", [0, 1])
user_input['Month_Nov'] = st.radio("Month_Nov", [0, 1])
user_input['Month_Oct'] = st.radio("Month_Oct", [0, 1])
user_input['Month_Sep'] = st.radio("Month_Sep", [0, 1])
user_input['OperatingSystems'] = st.number_input("Operating Systems", value=0)
user_input['Browser'] = st.number_input("Browser", value=0)
user_input['Region'] = st.number_input("Region", value=0)
user_input['TrafficType'] = st.number_input("Traffic Type", value=0)
user_input['VisitorType_Other'] = st.radio("Visitor Type_Other", [0, 1])
user_input['VisitorType_Returning_Visitor'] = st.radio("Visitor Type_Returning_Visitor", [0, 1])
user_input['Weekend_True'] = st.radio("Weekend_True", [0, 1])

st.warning(print_user_guide())

# Display user input
st.header("Session Information Entered")
st.write(user_input)

# Button to make prediction
if st.button("Predict"):
    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_input])

    # Make prediction
    prediction_label, probability = predict_user_input(model, scaler, label_encoder, user_df)

    # Display prediction
    st.header("Prediction")
    st.write(f"Predicted Revenue: {prediction_label}")
    st.write(f"Probability of Revenue: {probability:.2f}")
