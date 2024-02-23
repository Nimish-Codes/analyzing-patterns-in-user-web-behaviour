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

# Operating System dropdown
st.subheader("Select Operating System")
os_names = {1: "Windows", 2: "Macintosh", 3: "Linux", 4: "Other"}
user_input['OperatingSystems'] = st.selectbox("Operating System", list(os_names.values()))

# Browser dropdown
st.subheader("Select Browser")
browser_names = {1: "IE", 2: "Firefox", 3: "Chrome", 4: "Safari", 5: "Others"}
user_input['Browser'] = st.selectbox("Browser", list(browser_names.values()))

# Region dropdown
st.subheader("Select Region")
region_names = {1: "Region 1", 2: "Region 2", 3: "Region 3", 4: "Region 4", 5: "Region 5", 6: "Region 6", 7: "Region 7", 8: "Region 8", 9: "Region 9"}
user_input['Region'] = st.selectbox("Region", list(region_names.values()))

# Traffic Type dropdown
st.subheader("Select Traffic Type")
traffic_names = {1: "Type 1", 2: "Type 2", 3: "Type 3", 4: "Type 4", 5: "Type 5", 6: "Type 6", 7: "Type 7", 8: "Type 8", 9: "Type 9"}
user_input['TrafficType'] = st.selectbox("Traffic Type", list(traffic_names.values()))

# Weekend selection
user_input['Weekend'] = st.radio("Weekend", ["Yes", "No"])

# Convert Weekend to binary
user_input['Weekend'] = 1 if user_input['Weekend'] == "Yes" else 0

# Visitor Type
user_input['VisitorType_Returning_Visitor'] = st.radio("Visitor Type", ["Returning", "New"])  # Set as Returning Visitor

# Convert visitor type to binary
user_input['VisitorType_Returning_Visitor'] = 1 if user_input['VisitorType_Returning_Visitor'] == "Returning" else 0

# Display user input
st.header("Session Information Entered")
st.write(user_input)

# Function to print user guide
def print_user_guide():
    st.header("User Guide")
    st.write("Welcome! Please provide the following information for predicting online shopper's purchase intention:")
    st.write("\n1. Administrative: Number of pages related to account management or settings visited during your session.")
    st.write("   Example: Login page, account settings page.")
    st.write("2. Administrative Duration: Total time (in seconds) spent on administrative pages.")
    st.write("3. Informational: Number of pages containing general information visited during your session.")
    st.write("   Example: FAQ page, help guide.")
    st.write("4. Informational Duration: Total time (in seconds) spent on informational pages.")
    st.write("5. ProductRelated: Number of pages related to products or services visited during your session.")
    st.write("   Example: Product listing page, product detail page.")
    st.write("6. ProductRelated Duration: Total time (in seconds) spent on product-related pages.")
    st.write("7. BounceRates: Percentage of visitors who leave the website after viewing only one page.")
    st.write("   A high bounce rate may indicate visitors did not find the content engaging.")
    st.write("8. ExitRates: Percentage of visitors who leave the website from a specific page.")
    st.write("   It helps identify pages where users are most likely to exit the site.")
    st.write("9. PageValues: Average value of the pages visited during your session.")
    st.write("   Higher page values indicate pages contributing more to conversions.")
    st.write("10. SpecialDay: Closeness of the session to a special day (normalized).")
    st.write("    Example: Valentine's Day, Black Friday.")
    st.write("11. Month: Month of the session.")
    st.write("12. OperatingSystems: Operating system used during the session.")
    st.write("13. Browser: Web browser used during the session.")
    st.write("14. Region: Geographical region of the user.")
    st.write("15. TrafficType: Type of traffic source through which the user arrived at the website.")
    st.write("16. VisitorType: Type of visitor to the website (Returning Visitor).")
    st.write("17. Weekend: Whether the session occurred on a weekend.")

# Print user guide
print_user_guide()

# Validate user input
if 'VisitorType_Returning_Visitor' not in user_input:
    st.error("Please select visitor type.")

# Button to make prediction
if st.button("Predict"):
    if 'VisitorType_Returning_Visitor' in user_input:
        # Convert user input to DataFrame
        user_df = pd.DataFrame([user_input])

        # Make prediction
        prediction_label, probability = predict_user_input(model, scaler, label_encoder, user_df)

        # Display prediction
        st.header("Prediction")
        st.write(f"Predicted Revenue: {prediction_label}")
        st.write(f"Probability of Revenue: {probability:.2f}")
