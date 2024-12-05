import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('train.csv')
print(df.info())

# Define features and target variable
X = df[['GrLivArea', 'TotRmsAbvGrd', 'OverallQual', 'LotArea', 'YearBuilt']]
y = df['SalePrice']

# Drop rows with missing target values
df.dropna(subset=['SalePrice'], inplace=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)

print("\nEnter details for a new house:")
try:
    gr_liv_area = float(input("Enter Gross Living Area (in square feet): "))
    tot_rms_abv_grd = int(input("Enter Total Rooms Above Ground: "))
    overall_qual = int(input("Enter Overall Quality (1-10): "))
    lot_area = float(input("Enter Lot Area (in square feet): "))
    year_built = int(input("Enter Year Built: "))

    new_house = pd.DataFrame([[gr_liv_area, tot_rms_abv_grd, overall_qual, lot_area, year_built]],
                             columns=['GrLivArea', 'TotRmsAbvGrd', 'OverallQual', 'LotArea', 'YearBuilt'])
    
    new_house_scaled = scaler.transform(new_house)
    
    predicted_price = model.predict(new_house_scaled)
    print(f"\nPredicted price for the new house: ${predicted_price[0]:,.2f}")
except ValueError:
    print("Please enter valid numeric values.")
