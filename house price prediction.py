import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate some random data for demonstration purposes
np.random.seed(42)
data_size = 100
rooms = np.random.randint(1, 6, size=data_size)
square_feet = np.random.randint(800, 2500, size=data_size)
house_prices = 50_000 + 10_000 * rooms + 200 * square_feet + np.random.normal(0, 10_000, size=data_size)

# Create a DataFrame from the generated data
data = pd.DataFrame({'Rooms': rooms, 'SquareFeet': square_feet, 'Price': house_prices})

# Split the data into training and testing sets
X = data[['Rooms', 'SquareFeet']]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot the predictions against the actual prices
plt.scatter(X_test['SquareFeet'], y_test, color='black', label='Actual Prices')
plt.scatter(X_test['SquareFeet'], predictions, color='blue', label='Predicted Prices')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.legend()
plt.show()

