import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from urllib.request import urlretrieve

# URL of the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"

# Download the dataset
urlretrieve(url, "online_shoppers_intention.csv")

# Load dataset
data = pd.read_csv("online_shoppers_intention.csv")

# Drop rows with missing values
data.dropna(inplace=True)

def print_user_guide():
    print("Welcome! Please provide the following information for predicting online shopper's purchase intention:")
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

# Function to get user input and make predictions
def predict_user_input(model, scaler, label_encoder):
    print("Please enter the following details:")
    # Prompt user for input
    user_input = {}
    user_input['Administrative'] = int(input("Administrative: "))
    user_input['Administrative_Duration'] = float(input("Administrative Duration: "))
    user_input['Informational'] = int(input("Informational: "))
    user_input['Informational_Duration'] = float(input("Informational Duration: "))
    user_input['ProductRelated'] = int(input("Product Related: "))
    user_input['ProductRelated_Duration'] = float(input("Product Related Duration: "))
    user_input['BounceRates'] = float(input("Bounce Rates: "))
    user_input['ExitRates'] = float(input("Exit Rates: "))
    user_input['PageValues'] = float(input("Page Values: "))
    user_input['SpecialDay'] = float(input("Special Day: "))
    user_input['Month_Dec'] = int(input("Month_Dec (0 or 1): "))
    user_input['Month_Feb'] = int(input("Month_Feb (0 or 1): "))
    user_input['Month_Jul'] = int(input("Month_Jul (0 or 1): "))
    user_input['Month_June'] = int(input("Month_June (0 or 1): "))
    user_input['Month_Mar'] = int(input("Month_Mar (0 or 1): "))
    user_input['Month_May'] = int(input("Month_May (0 or 1): "))
    user_input['Month_Nov'] = int(input("Month_Nov (0 or 1): "))
    user_input['Month_Oct'] = int(input("Month_Oct (0 or 1): "))
    user_input['Month_Sep'] = int(input("Month_Sep (0 or 1): "))
    user_input['OperatingSystems'] = int(input("Operating Systems: "))
    user_input['Browser'] = int(input("Browser: "))
    user_input['Region'] = int(input("Region: "))
    user_input['TrafficType'] = int(input("Traffic Type: "))
    user_input['VisitorType_Other'] = int(input("Visitor Type_Other (0 or 1): "))
    user_input['VisitorType_Returning_Visitor'] = int(input("Visitor Type_Returning_Visitor (0 or 1): "))
    user_input['Weekend_True'] = int(input("Weekend_True (0 or 1): "))

    # Print user guide
    print_user_guide()

    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_input])

    # Scale user input
    user_scaled = scaler.transform(user_df)

    # Make prediction
    prediction = model.predict(user_scaled)
    probability = model.predict_proba(user_scaled)[:, 1]

    # Decode prediction
    prediction_label = label_encoder.inverse_transform(prediction)[0]
    print(f"\nPredicted Revenue: {prediction_label}")
    print(f"Probability of Revenue: {probability[0]}")

# Predict user input
predict_user_input(model, scaler, label_encoder)
