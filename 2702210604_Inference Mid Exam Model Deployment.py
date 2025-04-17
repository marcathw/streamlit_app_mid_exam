import pandas as pd
import pickle

def load_pickle(file):
    with open(file, "rb") as f:
        model = pickle.load(f)
    return model

def process_sample(sample_dict):
    ord_enc = {
        "type_of_meal_plan": {
            "Not Selected": 0,
            "Meal Plan 1": 1,
            "Meal Plan 2": 2,
            "Meal Plan 3": 3
        },
        "room_type_reserved": {
            "Room_Type 1": 1,
            "Room_Type 2": 2,
            "Room_Type 3": 3,
            "Room_Type 4": 4,
            "Room_Type 5": 5,
            "Room_Type 6": 6,
            "Room_Type 7": 7
        }
    }

    for col, mapping in ord_enc.items():
        sample_dict[col] = mapping[sample_dict[col]]

    market_segment = sample_dict["market_segment_type"]
    for cat in ["Aviation", "Complementary", "Corporate", "Offline", "Online"]:
        sample_dict[f"market_segment_type_{cat}"] = 1 if market_segment == cat else 0

    del sample_dict["market_segment_type"]
    return sample_dict

def predict(model, sample):
    sample_df = pd.DataFrame([sample_dict])
    result = model.predict(sample_df)
    return result[0]

if __name__ == "__main__":
    file = "2702210604_Pickle Mid Exam Model Deployment.pkl"
    model = load_pickle(file)
    sample_dict = {
        "no_of_adults": 2, 
        "no_of_children": 0, 
        "no_of_weekend_nights": 2, 
        "no_of_week_nights": 3, 
        "type_of_meal_plan": "Not Selected", 
        "required_car_parking_space": 0.0, 
        "room_type_reserved": "Room_Type 1", 
        "lead_time": 5, 
        "arrival_year": 2018, 
        "arrival_month": 11, 
        "arrival_date": 6, 
        "market_segment_type": "Complementary", 
        "repeated_guest": 1, 
        "no_of_previous_cancellations": 0, 
        "no_of_previous_bookings_not_canceled": 0, 
        "avg_price_per_room": 63.68, 
        "no_of_special_requests": 1
    }

    result = predict(model, process_sample(sample_dict))
    if result == 0:
        result = "Not Canceled"
    elif result == 1:
        result = "Canceled"
    print(f"Predicted booking_status = {result}")