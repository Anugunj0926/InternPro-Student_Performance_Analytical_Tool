import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
df = pd.read_csv("preprocessed_data.csv")

# Train/Test split
X = df.drop('G3_mat', axis=1)
y = df['G3_mat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf = RandomForestRegressor(random_state=42)
lr = LinearRegression()
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Predictions
rf_pred = rf.predict(X_test)
lr_pred = lr.predict(X_test)

# Evaluation
print("🔍 Random Forest R²:", r2_score(y_test, rf_pred))
print("🔍 Linear Regression R²:", r2_score(y_test, lr_pred))
print("📉 RMSE (RF):", mean_squared_error(y_test, rf_pred))
print("📉 RMSE (LR):", mean_squared_error(y_test, lr_pred))

# Plot Actual vs Predicted
plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_test, y=rf_pred, label="Random Forest", alpha=0.7)
sns.scatterplot(x=y_test, y=lr_pred, label="Linear Regression", alpha=0.7)
plt.plot([0, 20], [0, 20], 'r--')
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted G3 Score")
plt.legend()
plt.tight_layout()
#plt.savefig("actual_vs_predicted.png")
plt.show()

# import json

# Save feature columns
#with open("model_features.json", "w") as f:
#    json.dump(list(X.columns), f)

# Save the better model (assume RF for now)
# joblib.dump(rf, "rf_model.pkl")
# print("✅ Model saved as 'rf_model.pkl'")