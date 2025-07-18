import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib


# 1. Load Data
df = pd.read_csv("nasaa_with_area.csv")

# 2. Parse date column and extract time-based features
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['YearMonth'] = df['Date'].dt.to_period('M')

# 3. Create Debris Density if not present
if 'Debris_Density' not in df.columns:
    df['Debris_Density'] = df['Total_Debris'] / df['Estimated_Area_m2']

def classify_storm(value):
    if pd.isna(value):
        return "unknown"
    value = value.lower()
    if any(word in value for word in ['no storm', 'none', 'no significant', 'no recent', 'none noted']):
        return 'none'
    elif any(word in value for word in ['mild', 'clear', 'calm', 'sunny', 'nice', 'fair']):
        return 'mild'
    elif any(word in value for word in ['rain', 'shower', 'sprinkle']):
        return 'rain'
    elif any(word in value for word in ['storm', 'wind', 'high tide', 'hurricane', 'gale', 'surge']):
        return 'storm'
    elif any(word in value for word in ['unknown', 'u/n', 'not sure', 'unkown']):
        return 'unknown'
    else:
        return 'other'

df['Storm_Category'] = df['Storm_Activity'].apply(classify_storm)
le_storm = LabelEncoder()
df['Storm_Category_encoded'] = le_storm.fit_transform(df['Storm_Category'])

# 4. Optional: Encode Shoreline Name by mean target
shoreline_mean = df.groupby('Shoreline_Name')['Debris_Density'].mean().to_dict()
df['Shoreline_Mean'] = df['Shoreline_Name'].map(shoreline_mean)

# 5. Outlier Removal (IQR method)
Q1 = df['Debris_Density'].quantile(0.25)
Q3 = df['Debris_Density'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Debris_Density'] >= Q1 - 1.5 * IQR) & (df['Debris_Density'] <= Q3 + 1.5 * IQR)]

# 6. Define Features and Target
features = ['Total_Debris', 'Estimated_Area_m2', 'Year', 'Month','Storm_Category_encoded']
X = df[features]
y = df['Debris_Density']

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Pipeline with XGBoost
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
])

# 9. Train the model
pipeline.fit(X_train, y_train)

# 10. Predictions and Evaluation
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
print("Cross-Validated R²:", np.mean(cv_scores))

print(f"MSE: {mse}")
print(f"R² Score: {r2:.4f}")

# Add Date to test set for time-based plotting
X_test_plot = X_test.copy()
X_test_plot['Date'] = df.loc[X_test.index, 'Date']
X_test_plot['Actual_Density'] = y_test
X_test_plot['Predicted_Density'] = y_pred

# Sort by date for smooth plotting
X_test_plot = X_test_plot.sort_values('Date')

# Plot actual vs predicted
plt.plot(X_test_plot['Date'], X_test_plot['Actual_Density'], label='Actual', marker='o')
plt.plot(X_test_plot['Date'], X_test_plot['Predicted_Density'], label='Predicted', marker='x')
plt.title("Debris Density Over Time: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Debris Density (items/m²)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

joblib.dump(pipeline, r'debris_model.pkl')