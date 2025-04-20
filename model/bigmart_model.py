import numpy as np

class SimpleModel:
    """
    A simple prediction model for retail sales forecasting.
    """
    def __init__(self):
        # Base values for predictions
        self.base_value = 1000
        self.mrp_multiplier = 10
        self.outlet_type_effects = [500, 1500, 2500, 3500]  # Grocery Store, Supermarket Type1, Type2, Type3
        self.outlet_size_effects = [1500, 1000, 500]  # High, Medium, Small
        self.age_factor = -10  # Newer outlets perform better
        
        # Generate consistent random effects for outlet_ids
        np.random.seed(42)
        self.id_effects = np.random.normal(0, 200, 10)
        
    def predict(self, features):
        """
        Predict sales based on input features.
        
        Parameters:
        features (numpy.ndarray): Array of shape (n_samples, 5) with columns:
                                  [item_mrp, outlet_id, outlet_size, outlet_type, outlet_age]
        
        Returns:
        numpy.ndarray: Predicted sales values
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        predictions = []
        
        for i in range(features.shape[0]):
            item_mrp, outlet_id, outlet_size, outlet_type, outlet_age = features[i]
            
            # Base prediction
            pred = self.base_value
            
            # MRP effect (higher price, higher sales)
            pred += item_mrp * self.mrp_multiplier
            
            # Outlet type effect
            pred += self.outlet_type_effects[int(outlet_type)]
            
            # Outlet size effect
            pred += self.outlet_size_effects[int(outlet_size)]
            
            # Age effect (newer outlets perform better)
            pred += outlet_age * self.age_factor
            
            # Random variation for outlet_id
            pred += self.id_effects[int(outlet_id)]
            
            # Ensure prediction is positive
            predictions.append(max(pred, 100))
        
        return np.array(predictions)

# Create an instance of the model
model = SimpleModel()