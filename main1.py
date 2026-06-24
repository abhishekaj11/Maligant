import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report  
import joblib

#EDA
df =pd.read_csv("data/raw/tumor_data.csv")

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df['label'].value_counts())


plt.scatter(df[df.label==0]['tumor_size_cm'], df[df.label==0]['age'], label='Begin', color='blue', alpha=0.6)
plt.scatter(df[df.label==1]['tumor_size_cm'], df[df.label==1]['age'], label='Maligant', color='red', alpha=0.6)
plt.xlabel('tumor size(cm)')
plt.ylabel('age')
plt.legend()
plt.show()

#logistic_model

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self,z):
        return 1 /(1 + np.exp(-z))

    def fit(self, X, y):
        n, f =  X.shape
        self.w = np.zeros(f)
        self.b = 0
        self.losses = []    

        for epochs in range(self.epochs):
            z = X @ self.w + self.b
            yhat = self.sigmoid(z)

            loss = -np.mean(
                y * np.log(yhat + 1e-9) + 
                (1  - y) * np.log(1 - yhat + 1e-9)

            )
            self.losses.append(loss)

            err = yhat - y
            dw = X.T @ err/n
            db = err.mean()

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self,X):
        return self.sigmoid(X @ self.w + self.b)
    
    def predict (self ,X):
        return (self.predict_proba (X)>= 0.5).astype(int)
    
    #test
X = df[["tumor_size_cm", "age"]].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train mean: {X_train.mean(0).round(2)}")
print(f"Train std: {X_train.std(0).round(2)}")

# Train

model_scratch = LogisticRegression(lr=0.5, epochs=1500)
model_scratch.fit(X_train, y_train)

plt.figure(figsize=(8,4))
plt.plot(model_scratch.losses, color= "steelblue", linewidth="1.5")
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy loss')
plt.title('Training loss curve')
plt.grid(True, alpha=0.3)
plt.savefig('outputs/loss_curve.png')
print(" the loss curve was saved at ouputs/loss_curve.png")

print(f"Intial loss: {model_scratch.losses[0]:.4f}")
print(f"Final loss: {model_scratch.losses[-1]:.4f}")

#evalute

y_pred = model_scratch.predict(X_test)
y_pred_proba = model_scratch.predict_proba(X_test)

print("Accuracy", round(accuracy_score(y_test, y_pred),3))
print("Precision", round(precision_score(y_test, y_pred),3))
print("Recall", round(recall_score(y_test, y_pred),3))
print("F1 Score", round(f1_score(y_test, y_pred),3))
print()
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))

#predict

new_patients = np.array([
    [2.1, 30],
    [6.8, 69],
    [9.2, 49],
])

new_scaled = scaler.transform(new_patients)

probs = model_scratch.predict_proba(new_scaled)
preds = model_scratch.predict(new_scaled)

labels = ["Begin", "Maligant"]

for i, (prob, pred) in enumerate (zip(probs,preds)):
    print(f"Pateints {i+1}: {labels[pred]} ({prob*100:.1f}% Maligant)")

#final

joblib.dump(model_scratch, 'tumor_model.pkl')
joblib.dump(scaler, 'model_scaler.pkl')
print("model Saved")

model_loaded = joblib.load('tumor_model.pkl')
scaler_loaded = joblib.load('tumor_scaler.pkl')

patients =np.array ([[6.0, 60]])
patient_scaled = scaler_loaded.transform(patients)
prob = model_loaded.predict_proba(patient_scaled)[0]
print(f"Maligant Probabilty: {prob*100:.1f}%")

