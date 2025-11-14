import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv('sales_data.csv')
print("Data loaded:\n", data.head())

X = data[['Week']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("\nTest predictions:", preds.round())

joblib.dump(model, 'demand_model.pkl')
print("\nModel saved as demand_model.pkl")
