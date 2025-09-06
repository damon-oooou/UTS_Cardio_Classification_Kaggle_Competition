import pandas as pd
from assi3_MLP import mlp, scaler      # import both your fitted model and its scaler :contentReference[oaicite:0]{index=0}
from EDA import clean_data 

# 1. Load raw test set
test_df = pd.read_csv('test_kaggle_features.csv')

# 2. Apply the same filter
test_df['ap_hi'] = test_df['ap_hi'].clip(80, 200)
test_df['ap_lo'] = test_df['ap_lo'].clip(50, 130)

# 3. Map the binary features exactly as in clean_data()
for col in ['smoke','alco','active']:
    test_df[col] = test_df[col].map({'No': 0, 'Yes': 1})

# 4. Select the same feature columns
feature_cols = [
    'age','gender','height','weight',
    'ap_hi','ap_lo','cholesterol','gluc',
    'smoke','alco','active'
]
X_test = test_df[feature_cols]

X_test_scaled = scaler.transform(X_test)

# 6. Predict using your trained MLP
y_pred = mlp.predict(X_test_scaled)

# 7. Write submission
submission = pd.DataFrame({
    'id':    test_df['id'],
    'cardio': y_pred
})


submission['cardio'] = submission['cardio'].map({0: 'No', 1: 'Yes'})
submission.to_csv('submission.csv', index=False)



