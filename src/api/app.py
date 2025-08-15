from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import uuid
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

class HybridNetworkFaultDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.lstm_model = None
        self.is_trained = False
        self.sequence_length = 24
        self.threshold = 0.7
        self._initialize_demo_models()
        
    def _initialize_demo_models(self):
        np.random.seed(42)
        normal_data = np.random.randn(800, 7) * 0.5 + [50, 0.5, 99, 80, 60, 50, 0.1]
        anomaly_data = np.random.randn(200, 7) * 2 + [200, 5, 90, 30, 90, 80, 2]
        demo_data = np.vstack([normal_data, anomaly_data])
        
        self.scaler.fit(demo_data)
        self.isolation_forest.fit(self.scaler.transform(demo_data))
        self.lstm_model = self._build_lstm_model()
        self.is_trained = True
        print("✅ Hybrid ML models initialized")
    
    def _build_lstm_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 7)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        X_synthetic = np.random.randn(100, self.sequence_length, 7)
        y_synthetic = np.random.randint(0, 2, 100)
        model.fit(X_synthetic, y_synthetic, epochs=5, verbose=0, batch_size=16)
        return model
    
    def predict_fault(self, current_kpis, historical_sequence=None):
        features = np.array([
            current_kpis.get('latency', 0),
            current_kpis.get('packet_loss', 0),
            current_kpis.get('availability', 100),
            current_kpis.get('throughput', 100),
            current_kpis.get('cpu_utilization', 0),
            current_kpis.get('memory_usage', 0),
            current_kpis.get('error_rate', 0)
        ]).reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        isolation_score = self.isolation_forest.decision_function(features_scaled)[0]
        is_anomaly_if = self.isolation_forest.predict(features_scaled)[0] == -1
        
        if historical_sequence is not None and len(historical_sequence) >= self.sequence_length:
            sequence = np.array(historical_sequence[-self.sequence_length:]).reshape(1, self.sequence_length, 7)
            lstm_probability = float(self.lstm_model.predict(sequence, verbose=0)[0][0])
        else:
            sequence = np.random.randn(1, self.sequence_length, 7) * 0.1 + features
            lstm_probability = float(self.lstm_model.predict(sequence, verbose=0)[0][0])
        
        if_score_normalized = (isolation_score + 1) / 2
        combined_score = 0.6 * lstm_probability + 0.4 * (1 - if_score_normalized)
        
        rule_boost = 0.0
        if current_kpis.get('latency', 0) > 200:
            rule_boost += 0.2
        if current_kpis.get('packet_loss', 0) > 5.0:
            rule_boost += 0.3
        if current_kpis.get('availability', 100) < 95:
            rule_boost += 0.2
        
        combined_score = min(1.0, combined_score + rule_boost)
        
        return {
            'combined_score': combined_score,
            'lstm_probability': lstm_probability,
            'isolation_forest_score': isolation_score,
            'is_anomaly_isolation_forest': is_anomaly_if,
            'rule_boost': rule_boost
        }

detector = HybridNetworkFaultDetector()

def create_app():
    app = Flask(__name__)
    CORS(app)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    @app.route('/', methods=['GET'])
    def home():
        return jsonify({
            'service': 'Hybrid Network Fault Detection API',
            'version': '2.0.0',
            'status': 'running',
            'models': ['LSTM', 'Isolation Forest', 'Rule-based']
        })
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0',
            'service': 'Hybrid Network Fault Detection API',
            'models_loaded': {
                'lstm': detector.lstm_model is not None,
                'isolation_forest': detector.is_trained
            }
        })
    
    @app.route('/predict', methods=['POST'])
    def predict_fault():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            cell_id = data.get('cell_id', 'UNKNOWN')
            current_kpis = {
                'latency': float(data.get('latency', 0)),
                'packet_loss': float(data.get('packet_loss', 0)),
                'availability': float(data.get('availability', 100)),
                'throughput': float(data.get('throughput', 100)),
                'cpu_utilization': float(data.get('cpu_utilization', 0)),
                'memory_usage': float(data.get('memory_usage', 0)),
                'error_rate': float(data.get('error_rate', 0))
            }
            
            ml_result = detector.predict_fault(current_kpis)
            combined_score = ml_result['combined_score']
            
            if combined_score >= 0.9:
                risk_level = "CRITICAL"
                fault_predicted = True
            elif combined_score >= 0.7:
                risk_level = "HIGH"
                fault_predicted = True
            elif combined_score >= 0.5:
                risk_level = "MEDIUM"
                fault_predicted = True
            else:
                risk_level = "LOW"
                fault_predicted = False
            
            prediction = {
                'timestamp': datetime.now().isoformat(),
                'cell_id': cell_id,
                'prediction': {
                    'fault_predicted': fault_predicted,
                    'combined_risk_score': round(combined_score, 3),
                    'risk_level': risk_level
                },
                'ml_analysis': {
                    'lstm_probability': round(ml_result['lstm_probability'], 3),
                    'isolation_forest_score': round(ml_result['isolation_forest_score'], 3),
                    'is_anomaly': bool(ml_result['is_anomaly_isolation_forest']),
                    'rule_boost': round(ml_result['rule_boost'], 3)
                },
                'kpi_values': current_kpis
            }
            
            if fault_predicted and risk_level in ['CRITICAL', 'HIGH']:
                prediction['ticket_generated'] = {
                    'ticket_id': f"NFD-ML-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'description': f"Hybrid AI-predicted {risk_level} fault for {cell_id}"
                }
            
            return jsonify(prediction)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
