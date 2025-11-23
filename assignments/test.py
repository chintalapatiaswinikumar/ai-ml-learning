import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Load Train + Test Data
# -----------------------------
df_train = pd.read_csv("bike_train.csv")   # your training file
df_test = pd.read_csv("bike_test.csv")  # the test file you uploaded

# -----------------------------
# 2. Function for date transformations
# -----------------------------
def transform_date_features(df, data_type):
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')

    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year

    if data_type == 'train':
        df = df.drop(columns=['datetime', 'casual', 'registered'])
        X = df.drop(columns=['count'])
        y = df['count']
        return X, y
    else:
        X = df.drop(columns=['datetime'])
        return X
    

# Apply the same transformations
X_train_raw, y_train = transform_date_features(df_train,"train")
X_test_raw  = transform_date_features(df_test,"test")

# -----------------------------
# 3. One-Hot Encoding (FIT ON TRAIN ONLY)
# -----------------------------
categorical_cols = ['season', 'weather', 'month', 'weekday', 'hour', 'year']

ct = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# fit ONLY on train
X_train_encoded = ct.fit_transform(X_train_raw)

# transform test (no fit)
X_test_encoded = ct.transform(X_test_raw)

# Convert to DataFrame
feature_names = (
    ct.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist() +
    [col for col in X_train_raw.columns if col not in categorical_cols]
)

X_train_df = pd.DataFrame(X_train_encoded.toarray(), columns=feature_names)
X_test_df = pd.DataFrame(X_test_encoded.toarray(), columns=feature_names)


# ------------------------------------------
# 1. Create polynomial features (degree=2)
# ------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly2 = poly.fit_transform(X_train_df)
X_test_poly2 = poly.transform(X_test_df)

print("Original shape:", X_train_df.shape)
print("Poly degree 2 shape:", X_train_poly2.shape)

# -----------------------------
# 4. Scaling
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_poly2)
X_test_scaled = scaler.transform(X_test_poly2)


# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred

alphas = [10]

print("Ridge Regression Results")
print("-------------------------")

for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_scaled, y_train)

    y_pred = ridge.predict(X_test_scaled)
    y_pred

df_test['count'] = y_pred
df_test.to_csv('Submission.csv')

# def rmsle(y_true,y_pred):
#     return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

# x=rmsle(1,-11.8633)
# print(x)