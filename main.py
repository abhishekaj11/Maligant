

import numpy as np
import pandas as pd
import os

def generate_tumor_data(n_samples=200, random_state=42):
    rng = np.random.RandomState(random_state)

    # Benign tumors: smaller size, younger age
    n_benign = n_samples // 2
    benign_size = rng.normal(loc=2.5, scale=0.8, size=n_benign).clip(0.5, 6.0)
    benign_age  = rng.normal(loc=38,  scale=10,  size=n_benign).clip(20, 80)

    # Malignant tumors: larger size, older age
    n_malignant = n_samples // 2
    malig_size  = rng.normal(loc=6.0, scale=1.2, size=n_malignant).clip(2.0, 10.0)
    malig_age   = rng.normal(loc=62,  scale=10,  size=n_malignant).clip(30, 90)

    sizes  = np.concatenate([benign_size, malig_size])
    ages   = np.concatenate([benign_age,  malig_age])
    labels = np.array([0] * n_benign + [1] * n_malignant)

    # Shuffle
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

# --- EDA.py ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/raw/tumor_data.csv')

# Basic exploration
print(df.head())
print(df.shape)          # (rows, columns)
print(df.dtypes)         # feature types
print(df.isnull().sum()) # missing values
print(df['label'].value_counts())  # class balance

# Visualize
plt.scatter(df[df.label==0]['tumor_size_cm'], df[df.label==0]['age'],
            label='Benign', color='blue', alpha=0.6)
plt.scatter(df[df.label==1]['tumor_size_cm'], df[df.label==1]['age'],
            label='Malignant', color='red', alpha=0.6)
plt.xlabel('Tumor size (cm)')
plt.ylabel('Age')
plt.legend()
plt.show()

# --- model.py ---
class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n, f = X.shape
        self.w = np.zeros(f)     # weights
        self.b = 0               # bias
        self.losses = []

        for epoch in range(self.epochs):
            z    = X @ self.w + self.b
            yhat = self.sigmoid(z)

            # Binary cross-entropy loss
            loss = -np.mean(
                y * np.log(yhat + 1e-9) +
                (1 - y) * np.log(1 - yhat + 1e-9)
            )
            self.losses.append(loss)

            # Gradients
            err  = yhat - y
            dw   = X.T @ err / n
            db   = err.mean()

            # Update weights
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# --- test.py ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features (X) and labels (y)
X = df[['tumor_size_cm', 'age']].values   # shape (200, 2)
y = df['label'].values             # shape (200,)

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (mean=0, std=1)
# CRITICAL: fit on train only, transform both
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)      # no fit_transform here!

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train mean: {X_train.mean(0).round(2)}")
print(f"Train std:  {X_train.std(0).round(2)}")

# --- train.py ---
model_scratch = LogisticRegression(lr=0.5, epochs=500)
model_scratch.fit(X_train, y_train)

# Plot loss curve
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

# --- evaluvate.py ---
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

# --- predict.py ---
# New patients to classify
new_patients = np.array([
    [2.1, 34],  # small tumor, young
    [7.5, 68],  # large tumor, elderly
    [4.2, 52],  # medium, middle-aged
])

# IMPORTANT: use same scaler from training!
new_scaled = scaler.transform(new_patients)

probs = model_scratch.predict_proba(new_scaled)
preds = model_scratch.predict(new_scaled)

labels = ['Benign', 'Malignant']
for i, (prob, pred) in enumerate(zip(probs, preds)):
    print(f"Patient {i+1}: {labels[pred]} ({prob*100:.1f}% malignant)")

# --- final.py ---
import joblib

# Save both model AND scaler
joblib.dump(model_scratch,  'tumor_model.pkl')
joblib.dump(scaler, 'tumor_scaler.pkl')
print("Model saved.")

# --- Later, in production ---
model_loaded  = joblib.load('tumor_model.pkl')
scaler_loaded = joblib.load('tumor_scaler.pkl')

# Predict on a new patient
patient = np.array([[5.0, 60]])
patient_scaled = scaler_loaded.transform(patient)
prob = model_loaded.predict_proba(patient_scaled)[0]
print(f"Malignant probability: {prob*100:.1f}%")
