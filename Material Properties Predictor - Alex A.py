import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def predict_property(data, target_property, known_properties):
    """
    Predict a target material property based on known properties.
    
    Args:
    - data (pd.DataFrame): The dataset containing material properties.
    - target_property (str): The name of the property to predict.
    - known_properties (dict): A dictionary of known property values for prediction.
    
    Returns:
    - float: Predicted value of the target property.
    """
    # Ensure all columns are numeric
    data = data.apply(pd.to_numeric, errors='coerce')
    
    # Drop rows where the target property is NaN
    data = data.dropna(subset=[target_property])
    
    # Separate features and target
    features = data.drop(columns=[target_property])
    target = data[target_property]
    
    # Drop features not provided in known_properties
    features = features[list(known_properties.keys())]
    
    # Handle missing values in features
    features = features.fillna(features.mean())
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    
    # Prepare the input for prediction
    input_data = pd.DataFrame([known_properties])
    
    # Predict the target property
    predicted_value = model.predict(input_data)[0]
    
    return predicted_value


# Example Usage
if __name__ == "__main__":
    # Load the dataset
    file_path = 'Data - Modified.csv'  # Adjust this to your file path
    data = pd.read_csv(file_path)
    
    # 'Bhn', 'Su', 'Sy', 'E', 'mu', 'Ro', 'G', A5
    target_property = 'Bhn'
    known_properties = {
        'Su': 440,
        'Sy': 370,
        'E': 205000,
        'Ro': 7870,
        'G': 800000,
        'A5': 15,
        'mu': 0.29
    }
    
    predicted_value = predict_property(data, target_property, known_properties)
    print(f"The predicted value for {target_property} is: {predicted_value:.2f}")
