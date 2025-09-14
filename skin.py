#!/usr/bin/env python3
"""
skin.ml - Complete Skin Type Prediction Machine Learning Pipeline
==================================================================

This file contains everything needed for skin type prediction:
- Data loading and preprocessing
- Model training
- Model testing
- Model evaluation
- Model saving/loading
- Prediction functionality

Author: Skin Type Prediction System

"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class SkinTypePredictionModel:
    """Complete skin type prediction model class"""
    
    def __init__(self):
        """Initialize the model"""
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.categorical_encoders = {}
        self.feature_cols = ['age', 'gender', 'water_intake', 'weather', 'oiliness', 
                           'acne', 'tightness_after_wash', 'makeup_usage', 'flaking', 'redness_itchiness']
        self.categorical_cols = ['gender', 'weather', 'oiliness', 'acne', 
                               'tightness_after_wash', 'makeup_usage', 'flaking', 'redness_itchiness']
        
    def load_and_preprocess_data(self, filepath="Data.csv", sample_size=10000):
        """Load and preprocess the dataset"""
        print("🚀 Loading and preprocessing data...")
        
        try:
            # Load data
            print(f"📊 Loading data from {filepath}...")
            if sample_size:
                df = pd.read_csv(filepath, nrows=sample_size)
                print(f"✅ Loaded {len(df)} samples (using sample for efficiency)")
            else:
                df = pd.read_csv(filepath)
                print(f"✅ Loaded {len(df)} samples")
            
            # Rename column to match frontend expectations
            if 'water_intake_liters' in df.columns:
                df = df.rename(columns={'water_intake_liters': 'water_intake'})
                print("✅ Renamed 'water_intake_liters' to 'water_intake'")
            
            # Check data structure
            print(f"📋 Dataset shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            print(f"🔍 Missing values: {df.isnull().sum().sum()}")
            
            # Display skin type distribution
            print(f"🎯 Skin type distribution:")
            skin_type_counts = df['skin_type'].value_counts()
            for skin_type, count in skin_type_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {skin_type}: {count} ({percentage:.1f}%)")
            
            # Extract features and target
            X = df[self.feature_cols].copy()
            y = df['skin_type'].copy()
            
            return X, y, df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None, None
    
    def encode_features(self, X, y, fit_encoders=True):
        """Encode categorical features and target variable"""
        print("🔧 Encoding categorical features...")
        
        X_encoded = X.copy()
        
        # Encode categorical features
        for col in self.categorical_cols:
            if col in X.columns:
                if fit_encoders:
                    encoder = LabelEncoder()
                    X_encoded[col] = encoder.fit_transform(X[col])
                    self.categorical_encoders[col] = encoder
                    print(f"✅ Encoded {col}: {list(encoder.classes_)}")
                else:
                    if col in self.categorical_encoders:
                        # Handle unseen labels
                        X_encoded[col] = X[col].apply(
                            lambda x: x if x in self.categorical_encoders[col].classes_ 
                            else self.categorical_encoders[col].classes_[0]
                        )
                        X_encoded[col] = self.categorical_encoders[col].transform(X_encoded[col])
        
        # Encode target variable
        if fit_encoders:
            skin_type_encoder = LabelEncoder()
            y_encoded = skin_type_encoder.fit_transform(y)
            
            # Create label_encoders dict for app.py compatibility
            self.label_encoders = {
                'weather': self.categorical_encoders['weather'],
                'oiliness': self.categorical_encoders['oiliness'],
                'skin_type': skin_type_encoder
            }
            
            print(f"🎯 Skin types encoded: {list(skin_type_encoder.classes_)}")
            return X_encoded, y_encoded
        else:
            return X_encoded, y
    
    def scale_features(self, X_train, X_test=None, fit_scaler=True):
        """Scale features using StandardScaler"""
        print("📏 Scaling features...")
        
        if fit_scaler:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            print("✅ Fitted and transformed training data")
            
            if X_test is not None:
                X_test_scaled = self.scaler.transform(X_test)
                print("✅ Transformed test data")
                return X_train_scaled, X_test_scaled
            else:
                return X_train_scaled
        else:
            if self.scaler:
                return self.scaler.transform(X_train)
            else:
                print("❌ Scaler not fitted yet")
                return X_train
    
    def train_model(self, X_train, y_train):
        """Train the RandomForest model"""
        print("🤖 Training RandomForest model...")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1  # Use all available CPU cores
        )
        
        self.model.fit(X_train, y_train)
        print("✅ Model training completed")
        
        # Display feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("🔍 Top 5 Most Important Features:")
        for idx, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        print("📊 Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("📋 Classification Report:")
        skin_types = self.label_encoders['skin_type'].classes_
        print(classification_report(y_test, y_pred, target_names=skin_types))
        
        # Confusion matrix
        print("🔀 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=skin_types, columns=skin_types)
        print(cm_df)
        
        return accuracy, y_pred, y_pred_proba
    
    def save_model(self, model_dir="."):
        """Save model and all preprocessors"""
        print("💾 Saving model and preprocessors...")
        
        # Save model components
        joblib.dump(self.model, os.path.join(model_dir, "skin_type_model.pkl"), compress=3)
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.pkl"), compress=3)
        joblib.dump(self.categorical_encoders, os.path.join(model_dir, "categorical_encoders.pkl"), compress=3)
        joblib.dump(self.label_encoders, os.path.join(model_dir, "skin_type_encoder.pkl"), compress=3)
        
        print("✅ Model saved successfully!")
        print("📁 Saved files:")
        print("  - skin_type_model.pkl")
        print("  - scaler.pkl")
        print("  - categorical_encoders.pkl")
        print("  - skin_type_encoder.pkl")
    
    def load_model(self, model_dir="."):
        """Load trained model and preprocessors"""
        print("📂 Loading model and preprocessors...")
        
        try:
            self.model = joblib.load(os.path.join(model_dir, "skin_type_model.pkl"))
            self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
            self.categorical_encoders = joblib.load(os.path.join(model_dir, "categorical_encoders.pkl"))
            self.label_encoders = joblib.load(os.path.join(model_dir, "skin_type_encoder.pkl"))
            
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_single(self, user_data):
        """Predict skin type for a single user input"""
        if not self.model:
            print("❌ Model not loaded")
            return None
        
        try:
            # Convert user data to the required format
            processed_data = {}
            
            # Numeric values
            processed_data['age'] = int(user_data['age'])
            processed_data['water_intake'] = float(user_data['water_intake'])
            
            # Binary encodings (matching app.py logic)
            processed_data['gender'] = 1 if user_data['gender'] == 'female' else 0
            processed_data['acne'] = 1 if user_data['acne'] == 'yes' else 0
            processed_data['tightness_after_wash'] = 1 if user_data['tightness_after_wash'] == 'yes' else 0
            processed_data['flaking'] = 1 if user_data['flaking'] == 'yes' else 0
            processed_data['redness_itchiness'] = 1 if user_data['redness_itchiness'] == 'yes' else 0
            
            # Makeup usage encoding
            makeup_mapping = {'never': 0, 'rare': 1, 'frequent': 2}
            processed_data['makeup_usage'] = makeup_mapping.get(user_data['makeup_usage'], 0)
            
            # Categorical encodings using saved encoders
            processed_data['weather'] = self.label_encoders['weather'].transform([user_data['weather']])[0]
            processed_data['oiliness'] = self.label_encoders['oiliness'].transform([user_data['oiliness']])[0]
            
            # Create feature array
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
            
            # Scale features
            features_scaled = self.scaler.transform(feature_array)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)
            prediction_proba = self.model.predict_proba(features_scaled)
            
            # Convert prediction to skin type
            skin_type = self.label_encoders['skin_type'].inverse_transform(prediction)[0]
            confidence = prediction_proba[0][prediction[0]]
            
            # Get all probabilities
            skin_types = self.label_encoders['skin_type'].classes_
            probabilities = prediction_proba[0]
            all_scores = {skin_type: prob for skin_type, prob in zip(skin_types, probabilities)}
            
            return {
                'skin_type': skin_type,
                'confidence': float(confidence),
                'all_scores': all_scores
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def complete_pipeline(self, data_file="Data.csv", sample_size=10000):
        """Run the complete ML pipeline"""
        print("🚀 Starting Complete Skin Type Prediction Pipeline")
        print("=" * 60)
        
        # Step 1: Load and preprocess data
        X, y, df = self.load_and_preprocess_data(data_file, sample_size)
        if X is None:
            return False
        
        # Step 2: Encode features
        X_encoded, y_encoded = self.encode_features(X, y, fit_encoders=True)
        
        # Step 3: Split data
        print("📊 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        print(f"📈 Training samples: {len(X_train)}")
        print(f"📈 Testing samples: {len(X_test)}")
        
        # Step 4: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, fit_scaler=True)
        
        # Step 5: Train model
        self.train_model(X_train_scaled, y_train)
        
        # Step 6: Evaluate model
        accuracy, y_pred, y_pred_proba = self.evaluate_model(X_test_scaled, y_test)
        
        # Step 7: Save model
        self.save_model()
        
        # Step 8: Test with sample prediction
        print("🧪 Testing with sample prediction...")
        test_data = {
            'age': 25,
            'gender': 'female',
            'water_intake': 2.0,
            'weather': 'dry',
            'oiliness': 'high',
            'acne': 'yes',
            'tightness_after_wash': 'no',
            'makeup_usage': 'frequent',
            'flaking': 'no',
            'redness_itchiness': 'no'
        }
        
        result = self.predict_single(test_data)
        if result:
            print(f"🎯 Sample prediction: {result['skin_type']} ({result['confidence']:.3f})")
        
        print("=" * 60)
        print("🎉 Pipeline completed successfully!")
        print(f"📊 Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return True

def main():
    """Main function to run the complete pipeline"""
    print("🌟 Skin Type Prediction ML System")
    print("=" * 50)
    
    # Create model instance
    skin_model = SkinTypePredictionModel()
    
    # Run complete pipeline
    success = skin_model.complete_pipeline(
        data_file="Data.csv", 
        sample_size=10000  # Use 10k samples for efficiency
    )
    
    if success:
        print("All operations completed successfully!")
        print(" Model is ready for use in your web application!")
    else:
        print("❌ Pipeline failed!")

if __name__ == "__main__":
    # Set working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run main function
    main()