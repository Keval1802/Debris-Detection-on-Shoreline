import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone
from xgboost import XGBRegressor
import joblib

# 1. Load Dataset
df = pd.read_csv("nasaa_with_area.csv")

# 2. Create Target Variable
if 'Debris_Density' not in df.columns:
    df['Debris_Density'] = df['Total_Debris'] / df['Estimated_Area_m2']

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['YearMonth'] = df['Date'].dt.to_period('M')

# 3. Encode Storm Category

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

if 'Storm_Category' in df.columns:
    le = LabelEncoder()
    df['Storm_Category_encoded'] = le.fit_transform(df['Storm_Category'])
else:
    raise ValueError("Storm_Category column not found")

# 4. Handle Date Column
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
else:
    print("⚠️ Date column not found. Time-based plot will be skipped.")

# 5. Outlier Removal (IQR)
Q1 = df['Debris_Density'].quantile(0.25)
Q3 = df['Debris_Density'].quantile(0.75)
IQR = Q3 - Q1

df = df[
    (df['Debris_Density'] >= Q1 - 1.5 * IQR) &
    (df['Debris_Density'] <= Q3 + 1.5 * IQR)
]

# 6. Feature Selection
features = [
    'Total_Debris',
    'Estimated_Area_m2',
    'Year',
    'Month',
    'Storm_Category_encoded'
]

X = df[features]
y = df['Debris_Density']

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(
        random_state=42,
        objective='reg:squarederror'
    ))
])

# 9. Hyperparameter Grid
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.8, 1.0]
}

# 10. Grid Search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

# 11. Evaluation
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nTest MSE: {mse:.4f}")
print(f"Test R² Score: {r2:.4f}")

cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-Validated R²: {cv_scores.mean():.4f}")

# 12. Actual vs Predicted Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Debris Density")
plt.ylabel("Predicted Debris Density")
plt.title("Actual vs Predicted Debris Density")
plt.show()

# 13. Learning Curve
train_sizes = np.linspace(0.1, 1.0, 10)
train_scores, val_scores = [], []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for frac in train_sizes:
    size = int(len(X_train) * frac)
    X_sub = X_train.iloc[:size]
    y_sub = y_train.iloc[:size]

    model = clone(best_model)
    model.fit(X_sub, y_sub)

    train_scores.append(r2_score(y_sub, model.predict(X_sub)))
    val_scores.append(
        np.mean(cross_val_score(model, X_train, y_train, cv=kf, scoring='r2'))
    )

plt.figure(figsize=(8, 5))
plt.plot(train_sizes * len(X_train), train_scores, label="Training R²")
plt.plot(train_sizes * len(X_train), val_scores, label="Validation R²")
plt.xlabel("Training Set Size")
plt.ylabel("R² Score")
plt.title("Learning Curve")
plt.legend()
plt.show()

# 14. Time-Based Plot (Optional)
if 'Date' in df.columns:
    X_test_plot = X_test.copy()
    X_test_plot['Date'] = df.loc[X_test.index, 'Date']
    X_test_plot['Actual'] = y_test
    X_test_plot['Predicted'] = y_pred

    X_test_plot = X_test_plot.sort_values('Date')

    plt.figure(figsize=(10, 5))
    plt.plot(X_test_plot['Date'], X_test_plot['Actual'], label='Actual')
    plt.plot(X_test_plot['Date'], X_test_plot['Predicted'], label='Predicted')
    plt.xlabel("Date")
    plt.ylabel("Debris Density")
    plt.title("Debris Density Over Time")
    plt.legend()
    plt.show()

# 15. Save Model
joblib.dump(best_model, "debris_model.pkl")
print("\n Model saved as debris_model.pkl")
