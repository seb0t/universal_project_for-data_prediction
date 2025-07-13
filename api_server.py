#!/usr/bin/env python3
"""
ML Model API Server
Run this script to start the Flask API server for model predictions.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
import sys
sys.path.append('./')
from functions.data_utils import load_transformers, transform_new_data

# Configuration
MODELS_PATH = './models'
API_HOST = '127.0.0.1'
API_PORT = 5000

# Load model artifacts
print("🔄 Loading model artifacts...")
try:
    final_model = joblib.load(os.path.join(MODELS_PATH, 'final_model.pkl'))
    model_metadata = joblib.load(os.path.join(MODELS_PATH, 'model_metadata.pkl'))
    transformers = load_transformers(os.path.join(MODELS_PATH, 'transformers.pkl'))
    feature_names = model_metadata.get('features', [])
    print("✅ Model artifacts loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model artifacts: {e}")
    exit(1)

# Create Flask app
app = Flask(__name__)

def apply_transformers(feature_array, transformers_dict):
    """Apply transformers to feature array"""
    if not transformers_dict or not isinstance(transformers_dict, dict):
        return feature_array
    
    # Get column information
    numeric_cols = transformers_dict.get('numeric_cols', [])
    categorical_cols = transformers_dict.get('categorical_cols', [])
    
    # Convert to DataFrame for easier column handling
    feature_names = model_metadata.get('features', [])
    df = pd.DataFrame(feature_array, columns=feature_names)
    
    # Process numeric columns
    numeric_data = np.array([]).reshape(feature_array.shape[0], 0)
    if numeric_cols and len(numeric_cols) > 0:
        numeric_df = df[numeric_cols]
        
        # Apply numeric imputer
        if 'numeric_imputer' in transformers_dict:
            numeric_data = transformers_dict['numeric_imputer'].transform(numeric_df)
        else:
            numeric_data = numeric_df.values
        
        # Apply numeric scaler
        if 'numeric_scaler' in transformers_dict:
            numeric_data = transformers_dict['numeric_scaler'].transform(numeric_data)
    
    # Process categorical columns
    categorical_data = np.array([]).reshape(feature_array.shape[0], 0)
    if categorical_cols and len(categorical_cols) > 0:
        cat_df = df[categorical_cols]
        
        # Apply categorical imputer
        if 'categorical_imputer' in transformers_dict:
            cat_df_imputed = pd.DataFrame(
                transformers_dict['categorical_imputer'].transform(cat_df),
                columns=categorical_cols
            )
        else:
            cat_df_imputed = cat_df
        
        # Apply categorical encoder
        if 'categorical_encoder' in transformers_dict:
            categorical_data = transformers_dict['categorical_encoder'].transform(cat_df_imputed)
            if hasattr(categorical_data, 'toarray'):
                categorical_data = categorical_data.toarray()
        else:
            categorical_data = cat_df_imputed.values
    
    # Combine numeric and categorical data
    if numeric_data.shape[1] > 0 and categorical_data.shape[1] > 0:
        final_data = np.hstack([numeric_data, categorical_data])
    elif numeric_data.shape[1] > 0:
        final_data = numeric_data
    elif categorical_data.shape[1] > 0:
        final_data = categorical_data
    else:
        final_data = feature_array
    
    return final_data

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    import time
    return jsonify({
        'status': 'healthy',
        'message': 'ML API is running',
        'timestamp': time.time()
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        # Create a serializable version of the metadata
        info = {
            'model_name': model_metadata.get('model_name', 'RandomForest'),
            'model_type': model_metadata.get('model_type', 'RandomForestClassifier'),
            'problem_type': model_metadata.get('problem_type', 'multiclass_classification'),
            'features': model_metadata.get('features', []),
            'n_features': len(model_metadata.get('features', [])),
            'training_score': model_metadata.get('training_score', 'Not available'),
            'validation_score': model_metadata.get('validation_score', 'Not available'),
            'model_available': final_model is not None,
            'transformers_available': transformers is not None,
            'label_encoder_available': model_metadata.get('target_encoder') is not None
        }
        
        # Add test performance if available
        test_perf = model_metadata.get('test_performance', {})
        if test_perf and isinstance(test_perf, dict):
            # Only include serializable values
            info['test_performance'] = {}
            for key, value in test_perf.items():
                if isinstance(value, (int, float, str, bool, list)):
                    info['test_performance'][key] = value
                else:
                    info['test_performance'][key] = str(value)
        
        # Add label encoder classes if available
        if model_metadata.get('target_encoder') is not None:
            label_encoder = model_metadata['target_encoder']
            if hasattr(label_encoder, 'classes_'):
                info['target_classes'] = label_encoder.classes_.tolist()
            elif isinstance(label_encoder, dict):
                info['target_classes'] = list(label_encoder.keys())
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict_single():
    """Single prediction endpoint"""
    try:
        if final_model is None:
            return jsonify({'error': 'Model not available for predictions'}), 500
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Extract features
        features = data
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        # Get expected feature names
        expected_features = model_metadata.get('features', [])
        if not expected_features:
            return jsonify({'error': 'Model feature information not available'}), 500
            
        # Convert to array in the right order
        try:
            feature_array = np.array([[features.get(feat, 0) for feat in expected_features]])
        except Exception as e:
            return jsonify({'error': f'Error preparing features: {str(e)}'}), 400
        
        # Apply transformers if available
        if transformers is not None:
            try:
                feature_array = apply_transformers(feature_array, transformers)
            except Exception as e:
                return jsonify({'error': f'Error applying transformers: {str(e)}'}), 500
        
        # Make prediction
        prediction = final_model.predict(feature_array)[0]
        prediction_proba = None
        
        # Get prediction probabilities if available
        if hasattr(final_model, 'predict_proba'):
            try:
                proba = final_model.predict_proba(feature_array)[0]
                prediction_proba = proba.tolist()
            except:
                pass
        
        # Convert prediction to readable format
        if model_metadata.get('target_encoder') is not None:
            label_encoder = model_metadata['target_encoder']
            if hasattr(label_encoder, 'inverse_transform'):
                try:
                    prediction_label = label_encoder.inverse_transform([prediction])[0]
                except:
                    prediction_label = str(prediction)
            elif isinstance(label_encoder, dict):
                prediction_label = label_encoder.get(prediction, str(prediction))
            else:
                prediction_label = str(prediction)
        else:
            prediction_label = str(prediction)
        
        result = {
            'prediction': prediction_label,
            'raw_prediction': int(prediction) if isinstance(prediction, (int, np.integer)) else float(prediction),
            'features_used': expected_features,
            'n_features': len(expected_features)
        }
        
        if prediction_proba is not None:
            result['prediction_probabilities'] = prediction_proba
            if model_metadata.get('target_encoder') is not None and hasattr(model_metadata['target_encoder'], 'classes_'):
                result['class_labels'] = model_metadata['target_encoder'].classes_.tolist()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        if final_model is None:
            return jsonify({'error': 'Model not available for predictions'}), 500
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Extract batch of features
        batch_features = data
        if not batch_features:
            return jsonify({'error': 'No batch_features provided'}), 400
        
        # Get expected feature names
        expected_features = model_metadata.get('features', [])
        if not expected_features:
            return jsonify({'error': 'Model feature information not available'}), 500
        
        # Convert to array
        try:
            feature_arrays = []
            for features in batch_features:
                feature_array = [features.get(feat, 0) for feat in expected_features]
                feature_arrays.append(feature_array)
            feature_arrays = np.array(feature_arrays)
        except Exception as e:
            return jsonify({'error': f'Error preparing batch features: {str(e)}'}), 400
        
        # Apply transformers if available
        if transformers is not None:
            try:
                feature_arrays = apply_transformers(feature_arrays, transformers)
            except Exception as e:
                return jsonify({'error': f'Error applying transformers: {str(e)}'}), 500
        
        # Make predictions
        predictions = final_model.predict(feature_arrays)
        predictions_proba = None
        
        # Get prediction probabilities if available
        if hasattr(final_model, 'predict_proba'):
            try:
                predictions_proba = final_model.predict_proba(feature_arrays)
            except:
                pass
        
        # Convert predictions to readable format
        prediction_labels = []
        if model_metadata.get('target_encoder') is not None:
            label_encoder = model_metadata['target_encoder']
            for pred in predictions:
                if hasattr(label_encoder, 'inverse_transform'):
                    try:
                        label = label_encoder.inverse_transform([pred])[0]
                    except:
                        label = str(pred)
                elif isinstance(label_encoder, dict):
                    label = label_encoder.get(pred, str(pred))
                else:
                    label = str(pred)
                prediction_labels.append(label)
        else:
            prediction_labels = [str(pred) for pred in predictions]
        
        result = {
            'predictions': prediction_labels,
            'raw_predictions': [int(p) if isinstance(p, (int, np.integer)) else float(p) for p in predictions],
            'n_predictions': len(predictions),
            'features_used': expected_features,
            'n_features': len(expected_features)
        }
        
        if predictions_proba is not None:
            result['prediction_probabilities'] = predictions_proba.tolist()
            if model_metadata.get('target_encoder') is not None and hasattr(model_metadata['target_encoder'], 'classes_'):
                result['class_labels'] = model_metadata['target_encoder'].classes_.tolist()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting ML API Server...")
    print(f"📊 Model: {model_metadata.get('model_name', 'Unknown')}")
    print(f"🎯 Problem type: {model_metadata.get('problem_type', 'unknown')}")
    print(f"🔢 Features: {len(feature_names)}")
    print(f"🌐 Server: http://{API_HOST}:{API_PORT}")
    print("\n🔗 Available endpoints:")
    print(f"  GET  /health")
    print(f"  GET  /model_info") 
    print(f"  POST /predict")
    print(f"  POST /predict_batch")
    print("\n✅ Server ready!")
    
    app.run(host=API_HOST, port=API_PORT, debug=False)
