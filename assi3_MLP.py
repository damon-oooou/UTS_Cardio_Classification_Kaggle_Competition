import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load data
from EDA import clean_data
df = clean_data()

# 2. Features & target
X = df[['age','gender','height','weight','ap_hi','ap_lo',
        'cholesterol','gluc','smoke','alco','active']]
y = df['cardio_C']

# 3. Train/validation split (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y,stratify=y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train))
X_val = pd.DataFrame(scaler.transform(X_val))

# 4. Build MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,50),random_state=42, max_iter=200, early_stopping=True) 
# 5. Fit & predict
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_val)

print("Classification Report:\n", classification_report(y_val, y_pred)) #print report(Evaluate)
mlp_cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:\n", mlp_cm)

# 7.  5-fold CV on training set
cv_scores = cross_val_score(mlp, X_train, y_train,
                            cv=5, scoring='accuracy')
print(f"5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 8. Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=mlp_cm,
                              display_labels=['No','Yes'])
disp.plot()
plt.title("MLP Confusion Matrix")



#Roc curve and AUC value
from sklearn.metrics import roc_curve, roc_auc_score,RocCurveDisplay
y_probs = mlp.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_probs, multi_class="ovr", average="weighted")
print('The AUC is %.3f' % auc)

fpr, tpr, thresholds = roc_curve(y_val, y_probs)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot()
plt.plot([0,1],[0,1], linestyle='--', color='gray')  
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
