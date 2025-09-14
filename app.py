#import __name__
from flask import Flask ,request, jsonify, send_from_directory,render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import logging
import os

app = Flask(__name__, static_folder='statics', static_url_path='/static', template_folder='templates')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model components
model = None
scaler = None
label_encoders = None

def load_model_components():
    """Load the trained model and preprocessing components"""
    global model, scaler, label_encoders
    
    try:
        # Set the working directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Load model components
        model = joblib.load('skin_type_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('skin_type_encoder.pkl')
        # Note: categorical_encoders.pkl might not be used in current implementation
        
        logger.info("Model components loaded successfully")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def preprocess_input(data):
    """Preprocess user input to match training data format"""
    try:
        # Convert input data to the same format as training
        processed_data = {}
        
        # Direct numeric values
        processed_data['age'] = int(data['age'])
        processed_data['water_intake'] = float(data['water_intake'])
        
        # Binary encodings
        processed_data['gender'] = 1 if data['gender'] == 'female' else 0
        processed_data['acne'] = 1 if data['acne'] == 'yes' else 0
        processed_data['tightness_after_wash'] = 1 if data['tightness_after_wash'] == 'yes' else 0
        processed_data['flaking'] = 1 if data['flaking'] == 'yes' else 0
        processed_data['redness_itchiness'] = 1 if data['redness_itchiness'] == 'yes' else 0
        
        # Makeup usage encoding
        makeup_mapping = {'never': 0, 'rare': 1, 'frequent': 2}
        processed_data['makeup_usage'] = makeup_mapping.get(data['makeup_usage'], 0)
        
        # Categorical encodings using saved encoders
        processed_data['weather'] = label_encoders['weather'].transform([data['weather']])[0]
        processed_data['oiliness'] = label_encoders['oiliness'].transform([data['oiliness']])[0]
        
        # Create feature array in the same order as training data
        feature_array = np.array([[
            processed_data['age'],
            processed_data['gender'],
            processed_data['water_intake'],
            processed_data['weather'],
            processed_data['oiliness'],
            processed_data['acne'],
            processed_data['tightness_after_wash'],
            processed_data['makeup_usage'],
            processed_data['flaking'],
            processed_data['redness_itchiness']
        ]])
        
        return feature_array
        
    except Exception as e:
        logger.error(f"Error preprocessing input: {e}")
        raise

@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict skin type based on user input"""
    try:
        # Check if model is loaded
        if model is None or scaler is None or label_encoders is None:
            return jsonify({'error': 'Model not loaded', 'success': False}), 500
        
        # Get JSON data from request
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided', 'success': False}), 400
        
        # Validate required fields
        required_fields = ['age', 'gender', 'water_intake', 'weather', 'oiliness', 
                          'acne', 'tightness_after_wash', 'makeup_usage', 'flaking', 'redness_itchiness']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {missing_fields}', 'success': False}), 400
        
        # Preprocess input data
        features = preprocess_input(data)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)
        
        # Convert prediction to skin type
        skin_type = label_encoders['skin_type'].inverse_transform(prediction)[0]
        
        # Get all skin types and their probabilities
        skin_types = label_encoders['skin_type'].classes_
        probabilities = prediction_proba[0]
        
        # Create confidence scores dictionary
        confidence_scores = {}
        for skin_t, prob in zip(skin_types, probabilities):
            confidence_scores[skin_t] = float(prob)
        
        # Get the confidence for the predicted skin type
        confidence = float(probabilities[prediction[0]])
        
        # Prepare response
        response = {
            'skin_type': skin_type,
            'confidence': confidence,
            'all_scores': confidence_scores,
            'recommendations': get_recommendations(skin_type),
            'success': True
        }
        
        logger.info(f"Prediction successful: {skin_type} with {confidence:.3f} confidence")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

def get_recommendations(skin_type):
    """Get skincare recommendations based on skin type"""
    recommendations = {
        'normal': [
            'Use a gentle cleanser twice daily',
            'Apply a light, balanced moisturizer',
            'Use broad-spectrum sunscreen daily (SPF 30+)',
            'Maintain your current routine if it\'s working well',
            'Consider vitamin C serum for antioxidant protection'
        ],
        'dry': [
            'Use a cream-based or oil-based cleanser',
            'Apply a rich, emollient moisturizer twice daily',
            'Use a humidifier in your bedroom at night',
            'Avoid hot water when washing your face',
            'Consider hyaluronic acid serum for extra hydration'
        ],
        'oily': [
            'Use a gentle foaming cleanser twice daily',
            'Apply an oil-free, non-comedogenic moisturizer',
            'Consider salicylic acid (BHA) products for pore care',
            'Use blotting papers during the day to manage excess oil',
            'Try clay masks once or twice a week'
        ],
        'sensitive': [
            'Use fragrance-free, hypoallergenic products',
            'Always patch test new products before full application',
            'Avoid products with alcohol, strong fragrances, or harsh chemicals',
            'Use lukewarm water for cleansing',
            'Consider products with soothing ingredients like aloe vera or chamomile'
        ]
    }
    
    return recommendations.get(skin_type, recommendations['normal'])

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        model_status = model is not None and scaler is not None and label_encoders is not None
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_status,
            'timestamp': pd.Timestamp.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'unhealthy'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        return jsonify({
            'model_type': str(type(model).__name__),
            'features': ['age', 'gender', 'water_intake', 'weather', 'oiliness', 
                        'acne', 'tightness_after_wash', 'makeup_usage', 'flaking', 'redness_itchiness'],
            'skin_types': list(label_encoders['skin_type'].classes_),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model components on startup
    model_loaded = load_model_components()
    
    if not model_loaded:
        print("Warning: Model components could not be loaded. Prediction functionality will be limited.")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
