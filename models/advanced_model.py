import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

class AdvancedRetailModel:
    """
    Advanced model for retail sales prediction that combines RandomForest 
    and Gradient Boosting models.
    """
    def __init__(self):
        # Initialize model components
        self.rf_model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=12,
            min_samples_split=5,
            random_state=42
        )
        
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importances_ = None
        self.is_trained = False
    
    def preprocess_features(self, features):
        """
        Preprocess input features for prediction
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        if self.is_trained and hasattr(self, 'scaler') and self.scaler is not None:
            return self.scaler.transform(features)
        return features
    
    def train(self, X, y):
        """
        Train the model using the provided data
        
        Parameters:
        X (numpy.ndarray): Training features
        y (numpy.ndarray): Target variable (sales)
        """
        # Save feature names if provided as DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.rf_model.fit(X_scaled, y)
        self.gb_model.fit(X_scaled, y)
        
        # Combine feature importances (50% weight to each model)
        rf_importances = self.rf_model.feature_importances_
        gb_importances = self.gb_model.feature_importances_
        self.feature_importances_ = (rf_importances + gb_importances) / 2
        
        self.is_trained = True
        return self
    
    def predict(self, features):
        """
        Make predictions using the trained model
        
        Parameters:
        features (numpy.ndarray): Input features for prediction
        
        Returns:
        numpy.ndarray: Predicted sales values
        """
        if not self.is_trained:
            # Return a simple prediction if model is not trained
            return self._simple_prediction(features)
        
        # Preprocess features
        X_scaled = self.preprocess_features(features)
        
        # Make predictions with both models
        rf_pred = self.rf_model.predict(X_scaled)
        gb_pred = self.gb_model.predict(X_scaled)
        
        # Ensemble prediction (average of both models)
        predictions = (rf_pred + gb_pred) / 2
        
        return predictions
    
    def _simple_prediction(self, features):
        """
        Make simple predictions when the model is not trained
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        predictions = []
        
        for i in range(features.shape[0]):
            # Simple prediction logic based on features
            item_mrp = features[i][0]
            outlet_id = features[i][1]
            outlet_size = features[i][2]
            outlet_type = features[i][3]
            outlet_age = features[i][4]
            
            # Base prediction
            pred = 1000
            
            # MRP effect (higher price, higher sales)
            pred += item_mrp * 10
            
            # Outlet type effect - maps to feature index 3
            outlet_type_effects = [500, 1500, 2500, 3500]
            pred += outlet_type_effects[int(outlet_type)]
            
            # Outlet size effect - maps to feature index 2
            size_effects = [1500, 1000, 500]  # High, Medium, Small
            pred += size_effects[int(outlet_size)]
            
            # Age effect (newer outlets perform better)
            pred += outlet_age * -10
            
            # Random variation for outlet_id
            np.random.seed(int(42 + outlet_id))
            id_effect = np.random.normal(0, 200)
            pred += id_effect
            
            predictions.append(max(pred, 100))
        
        return np.array(predictions)
    
    def get_feature_importance(self):
        """
        Get feature importance from the model
        
        Returns:
        dict: Feature names and their importance scores
        """
        if not self.is_trained or self.feature_names is None:
            return {}
        
        # Create dictionary of feature importances
        importances = dict(zip(self.feature_names, self.feature_importances_))
        
        # Sort by importance
        return {k: v for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}
    
    def save(self, filepath):
        """
        Save the model to disk
        
        Parameters:
        filepath (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a saved model from disk
        
        Parameters:
        filepath (str): Path to the saved model
        
        Returns:
        AdvancedRetailModel: Loaded model instance
        """
        try:
            return joblib.load(filepath)
        except:
            # Return a new instance if loading fails
            print(f"Error loading model from {filepath}. Creating a new model.")
            return cls()

# Create the model
model = AdvancedRetailModel()