import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

# ----------------------------
# Load trained model & scaler
# ----------------------------
model = joblib.load("gradient_boosting_fare_model.joblib")
scaler = joblib.load("fare_scaler.joblib")

# ----------------------------
# Haversine Distance Function
# ----------------------------
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

# ----------------------------
# Streamlit UI Styling
# ----------------------------
st.markdown(
    """
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(to right, #9370DB, #D8BFD8);  /* Orange → Pink */
        color: white;
    }

    /* Center Predict button */
    div.stButton > button:first-child {
        display: block;
        margin: 0 auto;
        background-color: #6A0DAD;
        color: white;
        font-size: 18px;
        padding: 10px 25px;
        border-radius: 8px;
    }

    /* Fare output styling */
    .fare-output {
        color: white;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }

    /* Input labels */
    div.stNumberInput > label, div.stTimeInput > label, div.stSelectbox > label {
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Streamlit App Content
# ----------------------------
st.title("🚖 Taxi Fare Prediction App")
st.write("Estimate your NYC taxi fare before your trip")

# Location Inputs
pickup_lat = st.number_input("Pickup Latitude", value=40.7580)
pickup_lon = st.number_input("Pickup Longitude", value=-73.9855)
drop_lat = st.number_input("Dropoff Latitude", value=40.7128)
drop_lon = st.number_input("Dropoff Longitude", value=-74.0060)

# Passenger Count
passenger_count = st.selectbox("Passenger Count", [1, 2, 3, 4, 5, 6])

# Date input
pickup_date = st.date_input("Pickup Date", datetime.today())

# Time input
if 'pickup_time' not in st.session_state:
    st.session_state.pickup_time = datetime.now().time()
pickup_time = st.time_input("Pickup Time", value=st.session_state.pickup_time)
st.session_state.pickup_time = pickup_time
pickup_datetime = datetime.combine(pickup_date, pickup_time)

# Payment Type
payment_type = st.selectbox(
    "Payment Type",
    ["Credit Card", "Cash", "No Charge", "Dispute"]
)

# ----------------------------
# Predict Button
# ----------------------------
if st.button("Predict Fare 💵"):

    # Feature engineering
    trip_distance = haversine(pickup_lon, pickup_lat, drop_lon, drop_lat)
    pickup_hour = pickup_datetime.hour
    pickup_day_10 = pickup_datetime.weekday()
    is_weekend = int(pickup_day_10 >= 5)
    is_rush_hour = int(pickup_hour in [7,8,9,16,17,18])
    is_late_night = int(pickup_hour >= 22 or pickup_hour <= 5)
    am_pm_PM = int(pickup_hour >= 12)
    time_period_Morning = int(5 <= pickup_hour < 12)
    time_period_Afternoon = int(12 <= pickup_hour < 17)
    time_period_Evening = int(17 <= pickup_hour < 22)
    payment_type_2 = int(payment_type == "Cash")
    payment_type_3 = int(payment_type == "No Charge")
    payment_type_4 = int(payment_type == "Dispute")

    # Create input dataframe
    input_data = pd.DataFrame([{
        "trip_distance": trip_distance,
        "passenger_count": passenger_count,
        "pickup_hour": pickup_hour,
        "pickup_day_10": pickup_day_10,
        "is_rush_hour": is_rush_hour,
        "is_late_night": is_late_night,
        "is_weekend": is_weekend,
        "time_period_Morning": time_period_Morning,
        "time_period_Afternoon": time_period_Afternoon,
        "time_period_Evening": time_period_Evening,
        "am_pm_PM": am_pm_PM,
        "payment_type_2": payment_type_2,
        "payment_type_3": payment_type_3,
        "payment_type_4": payment_type_4
    }])

    # Scale numerical columns
    num_cols = ["trip_distance", "passenger_count", "pickup_hour"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display predicted fare in white, centered
    st.markdown(f'<div class="fare-output">💰 Estimated Total Fare: ${prediction:.2f}</div>', unsafe_allow_html=True)
