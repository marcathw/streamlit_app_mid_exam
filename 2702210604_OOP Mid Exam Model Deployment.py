# LIBRARIES
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle

class Preprocessor:
    def __init__(self, filepath = "Dataset_B_hotel.csv"):
        self.filepath = filepath
        self.data = None
        self.ohe_encoder = OneHotEncoder(sparse_output = False)
        self.catcols = ["type_of_meal_plan", "required_car_parking_space", "room_type_reserved", "market_segment_type", "repeated_guest", "booking_status"]
        self.bin_enc = {
            "booking_status": {
                "Not_Canceled": 0,
                "Canceled": 1
            }
        }
        self.ord_enc = {
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
        self.ohe_cols = ["market_segment_type"]

    def read_data(self):
        self.data = pd.read_csv(self.filepath)
        print("Data successfully read.")

    def drop_identifier(self):
        self.data = self.data.drop(columns = ["Booking_ID"])
        print("Identifier column dropped.")

    def divide(self):
        self.numcols = [col for col in self.data.columns if col not in self.catcols]
        print("Categorical and numerical column divided.")

    def check_duplicates(self):
        print("Total duplicates: ", self.data.duplicated().sum())

    def drop_duplicates(self):
        self.data = self.data.drop_duplicates().reset_index(drop = True)
        print("Duplicates dropped.")

    def check_missing_values(self):
        missing = pd.DataFrame({
            "column name": self.data.columns.tolist(),
            "missing values": self.data.isnull().sum().tolist(),
            "percentage": ((self.data.isnull().sum()) * 100 / len(self.data)).round(2).tolist()
        })
        print("Total missing values:")
        print(missing)

    def impute_missing_values(self):
        for col in self.data.columns[self.data.isna().any()]:
            if col in self.catcols:
                self.data.loc[self.data[col].isna(), col] = self.data[col].dropna().mode().iloc[0]
            elif col in self.numcols:
                self.data.loc[self.data[col].isna(), col] = self.data[col].dropna().median()
        print("Missing values imputed.")
    
    def encode(self):
        self.data = self.data.replace(self.bin_enc)
        self.data = self.data.replace(self.ord_enc)
        ohe_df = self.ohe_encoder.fit_transform(self.data[self.ohe_cols])
        ohe_df = pd.DataFrame(ohe_df, columns = self.ohe_encoder.get_feature_names_out(self.ohe_cols))
        self.data = pd.concat([self.data.drop(columns = self.ohe_cols), ohe_df], axis = 1)
        print("Encoding performed.")

    def define_x_y(self):
        x = self.data.drop(columns = ["booking_status"])
        y = self.data["booking_status"]
        return x, y, self.numcols

class Modelling:
    def __init__(self, x, y, numcols, test_size = 0.3, n_estimators = 100, max_depth = 11, min_samples_split = 5, criterion = "gini", random_state = 11):
        self.x = x
        self.y = y
        self.numcols = numcols
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = test_size, random_state = random_state)
        self.scaler = RobustScaler()
        self.model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, criterion = criterion, random_state = random_state)

    def scale(self):
        self.x_train[self.numcols] = self.scaler.fit_transform(self.x_train[self.numcols])
        self.x_test[self.numcols] = self.scaler.transform(self.x_test[self.numcols])
        print("Scaling performed.")

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate(self):
        self.y_pred = self.model.predict(self.x_test)
        print("Classification Report:")
        print(classification_report(self.y_test, self.y_pred))
        cm = confusion_matrix(self.y_test, self.y_pred)
        print("Confusion Matrix:")
        print(pd.DataFrame(cm))

    def model_save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

# ----------------------------------------------------
preprocessor = Preprocessor()
preprocessor.read_data()
preprocessor.drop_identifier()
preprocessor.divide()
preprocessor.check_duplicates()
preprocessor.drop_duplicates()
preprocessor.check_missing_values()
preprocessor.impute_missing_values()
preprocessor.encode()
x, y, numcols = preprocessor.define_x_y()

# ----------------------------------------------------
modelling = Modelling(x, y, numcols)
modelling.scale()
modelling.train()
modelling.evaluate()
modelling.model_save("2702210604_Pickle Mid Exam Model Deployment.pkl")