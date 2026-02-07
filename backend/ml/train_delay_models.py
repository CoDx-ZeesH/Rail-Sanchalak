import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from lightgbm import LGBMClassifier
from xgboost import XGBRegressor

# ======================
# LOAD DATA
# ======================
data = pd.read_csv("backend/datasets/train_delay_data.csv")

target_class = "delayed_flag"
target_reg = "delay_minutes"

X = data.drop([target_class, target_reg], axis=1)
y = data[target_class]

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# ======================
# PREPROCESSOR
# ======================
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# ======================
# SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# CLASSIFIER
# ======================
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LGBMClassifier())
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
plt.imshow(cm, cmap='gray_r')
plt.colorbar()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("graph/confusion_matrix.png", dpi=300)
plt.close()

# ======================
# FEATURE IMPORTANCE
# ======================
model = clf.named_steps['model']
importances = model.feature_importances_

plt.figure(figsize=(6,4))
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("graph/feature_importance.png", dpi=300)
plt.close()

# ======================
# REGRESSION (Delayed only)
# ======================
delayed_data = data[data[target_class] == 1]

X_reg = delayed_data.drop([target_class, target_reg], axis=1)
y_reg = delayed_data[target_reg]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor())
])

reg.fit(X_train_reg, y_train_reg)

y_pred_reg = reg.predict(X_test_reg)

rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2 = r2_score(y_test_reg, y_pred_reg)

print("RMSE:", rmse)
print("R2:", r2)

plt.figure(figsize=(6,4))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.plot([min(y_test_reg), max(y_test_reg)],
         [min(y_test_reg), max(y_test_reg)])
plt.title("Actual vs Predicted Delay")
plt.tight_layout()
plt.savefig("graph/regression_scatter.png", dpi=300)
plt.close()
