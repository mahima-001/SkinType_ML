
import pandas as pd
import joblib
import argparse

# Load the trained model and encoders
model = joblib.load('skin_type_model.joblib')
skin_type_encoder = joblib.load('skin_type_encoder.joblib')
categorical_encoders = joblib.load('categorical_encoders.joblib')
scaler = joblib.load('scaler.joblib')

def predict_skin_type(args):
    user_data = pd.DataFrame({
        'age': [args.age],
        'gender': [args.gender],
        'water_intake_liters': [args.water_intake],
        'weather': [args.weather],
        'oiliness': [args.oiliness],
        'acne': [args.acne],
        'tightness_after_wash': [args.tightness_after_wash],
        'makeup_usage': [args.makeup_usage],
        'flaking': [args.flaking],
        'redness_itchiness': [args.redness_itchiness]
    })

    # Preprocess the new data
    for col, encoder in categorical_encoders.items():
        # Handle unseen labels by using the first class as a default
        user_data[col] = user_data[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
        user_data[col] = encoder.transform(user_data[col])

    # Scale the data
    X_new_scaled = scaler.transform(user_data)

    # Make predictions
    predictions = model.predict(X_new_scaled)

    # Decode the predictions
    predicted_skin_types = skin_type_encoder.inverse_transform(predictions)

    # Print the results
    print(f"\nBased on your answers, your predicted skin type is: {predicted_skin_types[0]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict skin type based on user input.')
    parser.add_argument('--age', type=int, required=True, help='Your age')
    parser.add_argument('--gender', type=str, required=True, choices=['male', 'female'], help='Your gender')
    parser.add_argument('--water-intake', type=float, required=True, help='Daily water intake in liters')
    parser.add_argument('--weather', type=str, required=True, choices=['dry', 'cold', 'humid', 'hot'], help='Current weather')
    parser.add_argument('--oiliness', type=str, required=True, choices=['low', 'medium', 'high'], help='Skin oiliness')
    parser.add_argument('--acne', type=str, required=True, choices=['yes', 'no'], help='Do you have acne?')
    parser.add_argument('--tightness-after-wash', type=str, required=True, choices=['yes', 'no'], help='Skin tightness after washing')
    parser.add_argument('--makeup-usage', type=str, required=True, choices=['never', 'rare', 'frequent'], help='Makeup usage frequency')
    parser.add_argument('--flaking', type=str, required=True, choices=['yes', 'no'], help='Does your skin flake?')
    parser.add_argument('--redness-itchiness', type=str, required=True, choices=['yes', 'no'], help='Redness or itchiness')

    args = parser.parse_args()
    predict_skin_type(args)