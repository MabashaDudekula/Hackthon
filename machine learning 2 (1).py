#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Step 1: Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 2: Load or Generate the Dataset
# For demonstration purposes, we will create a simulated dataset
# Assume 'sensor_data', 'time_of_day', 'day_of_week', 'weather_conditions', and 'holiday' affect traffic flow
np.random.seed(42)

data_size = 1000
data = {
    'sensor_data': np.random.randint(100, 1000, data_size),  # IoT traffic sensors
    'time_of_day': np.random.randint(0, 24, data_size),  # 0 to 23 (hours)
    'day_of_week': np.random.randint(0, 7, data_size),  # 0 to 6 (Monday to Sunday)
    'weather_conditions': np.random.choice([0, 1], data_size),  # 0 = clear, 1 = bad weather
    'holiday': np.random.choice([0, 1], data_size),  # 0 = non-holiday, 1 = holiday
    'traffic_flow': np.random.randint(140, 1000, data_size)  # Simulated traffic flow
}

df = pd.DataFrame(data)

# Step 3: Preprocessing
# We will standardize the sensor data and other continuous features for the regression model

# Select features and target variable
X = df[['sensor_data', 'time_of_day', 'day_of_week', 'weather_conditions', 'holiday']]
y = df['traffic_flow']

# Standardize the feature variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Build and Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

# Step 8: Visualize the comparison between actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Traffic Flow")
plt.ylabel("Predicted Traffic Flow")
plt.title("Actual vs Predicted Traffic Flow")
plt.show()


# In[13]:


# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Step 2: Generate a sample dataset (for demonstration purposes)
# Features: population_density, waste_production_rate, recyclable_percentage, participation_rate
# Target: recycling_initiative (0 = Basic Collection, 1 = Recycling Plant, 2 = Composting, 3 = Waste-to-Energy)

np.random.seed(42)

# Generating a synthetic dataset
data_size = 1000
data = {
    'population_density': np.random.randint(100, 10000, data_size),  # people per square km
    'waste_production_rate': np.random.uniform(0.5, 5.0, data_size),  # tons per day
    'recyclable_percentage': np.random.uniform(10, 90, data_size),  # percentage of waste that is recyclable
    'participation_rate': np.random.uniform(0, 100, data_size),  # percentage of citizens participating in recycling
    'recycling_initiative': np.random.choice([0, 1, 2, 3], data_size)  # 0 = Basic, 1 = Recycling Plant, 2 = Composting, 3 = Waste-to-Energy
}

# Create a DataFrame
df = pd.DataFrame(data)

# Step 3: Preprocessing - split the data into features (X) and labels (y)
X = df[['population_density', 'waste_production_rate', 'recyclable_percentage', 'participation_rate']]
y = df['recycling_initiative']

# Standardize the features for better performance of the model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=['Basic Collection', 'Recycling Plant', 'Composting', 'Waste-to-Energy'])

# Print evaluation results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_rep)

# Step 8: Test the model with a new sample of input data
new_data = np.array([[2000, 3.5, 60, 75]])  # Population Density: 2000, Waste: 3.5 tons/day, 60% recyclable, 75% participation
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

# Mapping the predicted class back to the recycling initiative
initiative_mapping = {0: 'Basic Collection', 1: 'Recycling Plant', 2: 'Composting', 3: 'Waste-to-Energy'}
print(f"\nPredicted recycling initiative: {initiative_mapping[prediction[0]]}")


# In[14]:


# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 2: Load the dataset (for this example, we simulate a dataset)
# In a real-world scenario, replace this with actual air quality data
np.random.seed(42)

# Generating a synthetic dataset for demonstration purposes
data_size = 1000
data = {
    'traffic_level': np.random.uniform(0 
                                       
                                       
                                       , 100, data_size),  # Traffic congestion level
    'industrial_emissions': np.random.uniform(0, 200, data_size),  # Industrial emissions index
    'temperature': np.random.uniform(0, 40, data_size),  # Temperature in Celsius
    'humidity': np.random.uniform(10, 100, data_size),  # Humidity percentage
    'wind_speed': np.random.uniform(0, 10, data_size),  # Wind speed in km/h
    'air_quality_index': np.random.uniform(0, 500, data_size)  # Air Quality Index (AQI)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Step 3: Preprocessing - split the data into features (X) and target (y)
X = df[['traffic_level', 'industrial_emissions', 'temperature', 'humidity', 'wind_speed']]
y = df['air_quality_index']

# Step 4: Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Initialize and train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Step 8: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Step 9: Test the model with a new sample input (real-time data)
new_data = np.array([[60, 150, 25, 70, 5]])  # Example inputs: traffic, emissions, temperature, humidity, wind speed
new_data_scaled = scaler.transform(new_data)
predicted_aqi = model.predict(new_data_scaled)

# Print the predicted Air Quality Index
print(f"Predicted Air Quality Index (AQI): {predicted_aqi[0]:.2f}")


# In[ ]:




