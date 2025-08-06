import pandas as pd
import joblib

# Load test data
df = pd.read_csv("preprocessed_data.csv")
X = df.drop('G3_mat', axis=1)
y = df['G3_mat']

# Load model
model = joblib.load("rf_model.pkl")

# Predict and compare
y_pred = model.predict(X)
df['Predicted_G3'] = y_pred

# Save results
df[['G3_mat', 'Predicted_G3']].to_csv("model_predictions.csv", index=False)
print("✅ Model predictions saved to 'model_predictions.csv'")

# Documentation note (manual task)
print("📄 Remember to document your pipeline, GitHub usage, and Streamlit demo in a final README.md")
