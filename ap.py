import streamlit as st
import pickle
import pandas as pd
import os

# Load the pre-trained model
model = pickle.load(open("flight_rf.pkl", "rb"))

# Function to make predictions
def predict_flight_price(departure_time, arrival_time, total_stops, airline, source, destination):
    # Process the inputs (similar to your original code)
    Journey_day = int(pd.to_datetime(departure_time).day)
    Journey_month = int(pd.to_datetime(departure_time).month)

    Dep_hour = int(pd.to_datetime(departure_time).hour)
    Dep_min = int(pd.to_datetime(departure_time).minute)

    Arrival_hour = int(pd.to_datetime(arrival_time).hour)
    Arrival_min = int(pd.to_datetime(arrival_time).minute)

    # Calculate Duration
    dur_hour = abs(Arrival_hour - Dep_hour)
    dur_min = abs(Arrival_min - Dep_min)

    # Airline encoding (your original encoding dictionaries)
    airline_dict = {
        'Jet Airways': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'IndiGo': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Air India': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'Multiple carriers': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'SpiceJet': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'Vistara': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'GoAir': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'Multiple carriers Premium economy': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'Jet Airways Business': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'Vistara Premium economy': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'Trujet': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }
    
    airline_encoded = airline_dict.get(airline, [0] * 11)

    # Source encoding
    source_dict = {
        'Delhi': [1, 0, 0, 0],
        'Kolkata': [0, 1, 0, 0],
        'Mumbai': [0, 0, 1, 0],
        'Chennai': [0, 0, 0, 1]
    }

    source_encoded = source_dict.get(source, [0] * 4)

    # Destination encoding
    destination_dict = {
        'Cochin': [1, 0, 0, 0, 0],
        'Delhi': [0, 1, 0, 0, 0],
        'New Delhi': [0, 0, 1, 0, 0],
        'Hyderabad': [0, 0, 0, 1, 0],
        'Kolkata': [0, 0, 0, 0, 1]
    }

    destination_encoded = destination_dict.get(destination, [0] * 5)

    # Prepare feature array for prediction
    features = [
        total_stops,
        Journey_day,
        Journey_month,
        Dep_hour,
        Dep_min,
        Arrival_hour,
        Arrival_min,
        dur_hour,
        dur_min
    ] + airline_encoded + source_encoded + destination_encoded

    # Make the prediction
    prediction = model.predict([features])
    return round(prediction[0], 2)

# Function to save the data to a CSV file
def save_to_csv(data, file_name='flight_data.csv'):
    # If the CSV doesn't exist, create it with headers
    if not os.path.exists(file_name):
        df = pd.DataFrame(data, columns=[
            'Departure Time', 'Arrival Time', 'Total Stops', 'Airline', 'Source', 'Destination', 'Predicted Price'
        ])
        df.to_csv(file_name, index=False)
    else:
        # Append to existing CSV
        df_existing = pd.read_csv(file_name)
        df_new = pd.DataFrame(data, columns=[
            'Departure Time', 'Arrival Time', 'Total Stops', 'Airline', 'Source', 'Destination', 'Predicted Price'
        ])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(file_name, index=False)

# Streamlit App
def main():
    st.title("Flight Price Prediction")

    # Input fields
    departure_time = st.date_input("Date of Journey")
    departure_hour = st.number_input("Departure Hour", 0, 23, 0)
    departure_minute = st.number_input("Departure Minute", 0, 59, 0)
    arrival_time = st.date_input("Date of Arrival")
    arrival_hour = st.number_input("Arrival Hour", 0, 23, 0)
    arrival_minute = st.number_input("Arrival Minute", 0, 59, 0)

    total_stops = st.selectbox("Total Stops", options=[0, 1, 2, 3])

    airline = st.selectbox("Airline", options=[
        'Jet Airways', 'IndiGo', 'Air India', 'Multiple carriers',
        'SpiceJet', 'Vistara', 'GoAir', 'Multiple carriers Premium economy',
        'Jet Airways Business', 'Vistara Premium economy', 'Trujet'
    ])

    source = st.selectbox("Source", options=['Delhi', 'Kolkata', 'Mumbai', 'Chennai'])
    destination = st.selectbox("Destination", options=[
        'Cochin', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata'
    ])

    # Button to predict
    if st.button("Predict"):
        dep_time = pd.to_datetime(f"{departure_time} {departure_hour}:{departure_minute}")
        arr_time = pd.to_datetime(f"{arrival_time} {arrival_hour}:{arrival_minute}")

        price = predict_flight_price(dep_time, arr_time, total_stops, airline, source, destination)
        
        st.success(f"Your Flight price is Rs. {price}")

        # Save the input data and prediction to CSV
        data = [[dep_time, arr_time, total_stops, airline, source, destination, price]]
        save_to_csv(data)

        st.write("Data saved to 'flight_data.csv'")

if __name__ == "__main__":
    main()
