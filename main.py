

import numpy as np
import pandas as pd
import os

def generate_tumor_data(n_samples=200, random_state=42):
    rng = np.random.RandomState(random_state)

    n_benign = n_samples // 2
    benign_size = rng.normal(loc=2.0, scale=0.3, size=n_benign).clip(0.5, 6.0)
    benign_age  = rng.normal(loc=30,  scale=5,  size=n_benign).clip(20, 80)

    n_malignant = n_samples // 2
    malig_size  = rng.normal(loc=7.0, scale=0.5, size=n_malignant).clip(2.0, 10.0)
    malig_age   = rng.normal(loc=70,  scale=5,  size=n_malignant).clip(30, 90)

    sizes  = np.concatenate([benign_size, malig_size])
    ages   = np.concatenate([benign_age,  malig_age])
    labels = np.array([0] * n_benign + [1] * n_malignant)

    
    idx = rng.permutation(n_samples)
    df = pd.DataFrame({
        'tumor_size_cm': sizes[idx].round(1),
        'age':           ages[idx].astype(int),
        'label':         labels[idx]
    })
    return df


if __name__ == '__main__':
    os.makedirs('data/raw', exist_ok=True)
    df = generate_tumor_data(n_samples=200)
    df.to_csv('data/raw/tumor_data.csv', index=False)
    print(f"Dataset saved: {df.shape[0]} rows")
    print(df['label'].value_counts().to_string())

# EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('data/raw/tumor_data.csv')


print(df.head())
print(df.shape)          
print(df.dtypes)         
print(df.isnull().sum()) 
print(df['label'].value_counts())  


plt.scatter(df[df.label==0]['tumor_size_cm'], df[df.label==0]['age'],
            label='Benign', color='blue', alpha=0.6)
plt.scatter(df[df.label==1]['tumor_size_cm'], df[df.label==1]['age'],
            label='Malignant', color='red', alpha=0.6)
plt.xlabel('Tumor size (cm)')
plt.ylabel('Age')
plt.legend()
plt.show()

# model
class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n, f = X.shape
        self.w = np.zeros(f)     
        self.b = 0               
        self.losses = []

        for epoch in range(self.epochs):
            z    = X @ self.w + self.b
            yhat = self.sigmoid(z)

            
            loss = -np.mean(
                y * np.log(yhat + 1e-9) +
                (1 - y) * np.log(1 - yhat + 1e-9)
            )
            self.losses.append(loss)

            
            err  = yhat - y
            dw   = X.T @ err / n
            db   = err.mean()

            
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X = df[['tumor_size_cm', 'age']].values  
y = df['label'].values             


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)      

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train mean: {X_train.mean(0).round(2)}")
print(f"Train std:  {X_train.std(0).round(2)}")

# train
model_scratch = LogisticRegression(lr=0.5, epochs=1500)
model_scratch.fit(X_train, y_train)


plt.figure(figsize=(8, 4))
plt.plot(model_scratch.losses, color='steelblue', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('Binary cross-entropy loss')
plt.title('Training loss curve')
plt.grid(True, alpha=0.3)
plt.savefig('outputs/loss_curve.png')
print("Loss curve saved to outputs/loss_curve.png")

print(f"Initial loss: {model_scratch.losses[0]:.4f}")
print(f"Final loss:   {model_scratch.losses[-1]:.4f}")

# evaluvate
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
)

y_pred      = model_scratch.predict(X_test)
y_pred_prob = model_scratch.predict_proba(X_test)

print("Accuracy: ",  round(accuracy_score(y_test,  y_pred), 3))
print("Precision:",  round(precision_score(y_test, y_pred), 3))
print("Recall:   ",  round(recall_score(y_test,    y_pred), 3))
print("F1 Score: ",  round(f1_score(y_test,        y_pred), 3))
print()
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))

#  predict
new_patients = np.array([
    [2.1, 34],  
    [7.5, 68], 
    [4.2, 52],  


new_scaled = scaler.transform(new_patients)

probs = model_scratch.predict_proba(new_scaled)
preds = model_scratch.predict(new_scaled)

labels = ['Benign', 'Malignant']
for i, (prob, pred) in enumerate(zip(probs, preds)):
    print(f"Patient {i+1}: {labels[pred]} ({prob*100:.1f}% malignant)")

# final
import joblib

joblib.dump(model_scratch,  'tumor_model.pkl')
joblib.dump(scaler, 'tumor_scaler.pkl')
print("Model saved.")


model_loaded  = joblib.load('tumor_model.pkl')
scaler_loaded = joblib.load('tumor_scaler.pkl')


patient = np.array([[5.0, 60]])
patient_scaled = scaler_loaded.transform(patient)
prob = model_loaded.predict_proba(patient_scaled)[0]
print(f"Malignant probability: {prob*100:.1f}%")
