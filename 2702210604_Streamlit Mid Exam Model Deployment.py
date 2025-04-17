import streamlit as st
import joblib
import numpy as np
from datetime import date

model = joblib.load("2702210604_Pickle Mid Exam Model Deployment.pkl")

def one_hot_encode(sample_dict, column, categories):
    value = sample_dict[column]
    for cat in categories:
        sample_dict[f"{column}_{cat}"] = 1 if value == cat else 0
    del sample_dict[column]
    return sample_dict

def make_prediction(features):
    input = np.array(features).reshape(1, -1)
    prediction = model.predict(input)
    return prediction[0]

def main():
    st.title("Mid Exam Model Deployment: Streamlit")

    meal_plans = {
        "Not Selected": 0,
        "Meal Plan 1": 1,
        "Meal Plan 2": 2,
        "Meal Plan 3": 3
    }

    room_types = {
        "Room Type 1": 1,
        "Room Type 2": 2,
        "Room Type 3": 3,
        "Room Type 4": 4,
        "Room Type 5": 5,
        "Room Type 6": 6,
        "Room Type 7": 7
    }

    market_segments = ["Aviation", "Complementary", "Corporate", "Offline", "Online"]    

    no_of_adults = st.slider("Number of Adults", min_value = 0, max_value = 10, value = 0, step = 1)
    no_of_children = st.slider("Number of Children", min_value = 0, max_value = 10, value = 0, step = 1)
    no_of_weekend_nights = st.slider("Number of Weekend Nights", min_value = 0, max_value = 20, value = 0, step = 1)
    no_of_week_nights = st.slider("Number of Weekday Nights", min_value = 0, max_value = 20, value = 0, step = 1)
    type_of_meal_plan = st.selectbox("Type of Meal Plan", list(meal_plans.keys()))
    required_car_parking_space = st.radio("Required Car Parking Space?", ["Yes", "No"])
    room_type_reserved = st.selectbox("Type of Room", list(room_types.keys()))
    lead_time = st.number_input("Lead Time [0 - 600]", min_value = 0, max_value = 600, value = 0, step = 1)
    arrival_date_input = st.date_input("Arrival Date", min_value = date(2015, 1, 1), max_value = date(2025, 12, 31), value = date(2015, 1, 1))
    market_segment_type = st.selectbox("Market Segment Type", market_segments)
    repeated_guest = st.radio("Repeated Guest?", ["Yes", "No"])
    no_of_previous_cancellations = st.slider("Number of Previous Cancellations", min_value = 0, max_value = 20, value = 0, step = 1)
    no_of_previous_bookings_not_canceled = st.slider("Number of Previous Bookings Not Cancelled", min_value = 0, max_value = 100, value = 0, step = 1)
    avg_price_per_room = st.number_input("Average Room Price [0.00 - 1000.00]", min_value = 0.0, max_value = 1000.0, value = 0.0, step = 0.01, format = "%.2f")
    no_of_special_requests = st.slider("Number of Special Requests", min_value = 0, max_value = 10, value = 0, step = 1)

    arrival_year = arrival_date_input.year
    arrival_month = arrival_date_input.month
    arrival_day = arrival_date_input.day
    
    if st.button("Make Prediction"):
        sample_dict = {
            "type_of_meal_plan": meal_plans[type_of_meal_plan],
            "room_type_reserved": room_types[room_type_reserved],
            "required_car_parking_space": 1 if required_car_parking_space == "Yes" else 0,
            "repeated_guest": 1 if repeated_guest == "Yes" else 0,
            "market_segment_type": market_segment_type
        }
        sample_dict = one_hot_encode(sample_dict, "market_segment_type", market_segments)

        
        features = [no_of_adults, 
                    no_of_children, 
                    no_of_weekend_nights, 
                    no_of_week_nights, 
                    sample_dict["type_of_meal_plan"], 
                    sample_dict["required_car_parking_space"], 
                    sample_dict["room_type_reserved"], 
                    lead_time, 
                    arrival_year, 
                    arrival_month, 
                    arrival_day, 
                    sample_dict["market_segment_type_Aviation"], 
                    sample_dict["market_segment_type_Complementary"],
                    sample_dict["market_segment_type_Corporate"],
                    sample_dict["market_segment_type_Offline"],
                    sample_dict["market_segment_type_Online"],
                    sample_dict["repeated_guest"], 
                    no_of_previous_cancellations, 
                    no_of_previous_bookings_not_canceled, 
                    avg_price_per_room, 
                    no_of_special_requests]

        result = make_prediction(features)
        if result == 0:
            result = "Not Cancelled"
        elif result == 1:
            result = "Canceled"
        st.success(f"The predicted booking status is: {result}")

if __name__ == "__main__":
    main()