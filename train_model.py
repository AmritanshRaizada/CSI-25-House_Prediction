import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import joblib

# Load preprocessed training data
train = pd.read_csv('train_preprocessed.csv')

# Split into features and target
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# Train-test split (just for selection and validation purposes)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Feature selection using LassoCV
lasso = LassoCV(cv=5).fit(X_train, y_train)

# Select top features based on median importance
sfm = SelectFromModel(lasso, prefit=True, threshold='median')
X_train_selected = sfm.transform(X_train)

# 2. Final model training using Linear Regression
final_model = LinearRegression()
final_model.fit(X_train_selected, y_train)

# 3. Save model and selected features
selected_features = X_train.columns[sfm.get_support()].tolist()

joblib.dump(final_model, 'house_model.pkl')
joblib.dump(selected_features, 'selected_features.pkl')

print("✅ Model trained and saved as 'house_model.pkl'")
print(f"🔢 Selected features: {selected_features}")
