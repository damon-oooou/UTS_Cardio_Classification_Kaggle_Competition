import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (classification_report,confusion_matrix,ConfusionMatrixDisplay)
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

# 5. Train SVM classifier
svm= SVC(
    C=1.0,
    probability=True,
    random_state=42
)
svm.fit(X_train, y_train)

# 6. Predict & hold-out evaluation
y_pred = svm.predict(X_val)
print("Classification Report:\n", classification_report(y_val, y_pred))

svm_cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:\n", svm_cm)


# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=svm_cm, display_labels=['No','Yes'])
disp.plot(cmap='Blues')
plt.title("SVM Confusion Matrix")



#Roc curve and AUC value
from sklearn.metrics import roc_curve, roc_auc_score,RocCurveDisplay
y_probs = svm.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_probs, multi_class="ovr", average="weighted")
print('The AUC is %.3f' % auc)

fpr, tpr, thresholds = roc_curve(y_val, y_probs)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot()
plt.plot([0,1],[0,1], linestyle='--', color='gray')  
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()