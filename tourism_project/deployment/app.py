import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Load the saved model from the Hugging Face model hub
model_path = hf_hub_download(repo_id="samuelrego/tourism_package_model", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Purchase Prediction
st.title("Wellness Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase the **Wellness Tourism Package**
based on their profile and interaction data.
Please enter the customer details below to get a prediction.
""")

# Get user inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
num_followups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
num_trips = st.number_input("Number of Trips (per year)", min_value=1, max_value=20, value=2)
passport = st.selectbox("Passport", [0, 1])
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
own_car = st.selectbox("Own Car", [0, 1])
num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=5000, max_value=100000, value=20000, step=1000)

# Save inputs into a DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': num_person_visiting,
    'NumberOfFollowups': num_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])

# Predict button
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.subheader("Prediction Result:")
        st.success(f"The customer is **likely to purchase** the Wellness Tourism Package! (Confidence: {probability[1]:.2%})")
    else:
        st.subheader("Prediction Result:")
        st.warning(f"The customer is **unlikely to purchase** the Wellness Tourism Package. (Confidence: {probability[0]:.2%})")
