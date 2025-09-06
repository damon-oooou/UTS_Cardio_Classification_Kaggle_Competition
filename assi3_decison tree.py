import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# 1. Load data
from EDA import clean_data
df = clean_data()



X = df[['age','gender','height','weight','ap_hi','ap_lo',
        'cholesterol','gluc','smoke','alco','active']]
y = df['cardio_C']


# 2. Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 3. Train
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# 4. Predict
y_pred = dt.predict(X_val)
y_pred

print(classification_report(y_val, y_pred))

aaaa=confusion_matrix(y_val, y_pred)
print(aaaa)
display_M = ConfusionMatrixDisplay(
    confusion_matrix=aaaa,
    display_labels=dt.classes_    # or ['No','Yes']
)
display_M.plot(cmap='Blues')
plt.title("Confusion Matrix")



#Roc curve and AUC value
from sklearn.metrics import roc_curve, roc_auc_score,RocCurveDisplay
y_probs = dt.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_probs, multi_class="ovr", average="weighted")
print('The AUC is %.3f' % auc)

fpr, tpr, thresholds = roc_curve(y_val, y_probs)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot()
plt.plot([0,1],[0,1], linestyle='--', color='gray')  
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()



