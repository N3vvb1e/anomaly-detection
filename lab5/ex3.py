import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

data = loadmat("shuttle.mat")
X = data["X"]
y = data["y"].ravel()

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Contamination rate (dataset): {np.mean(y == 1):.4f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

print(f"Train set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)

den = (X_max - X_min)
den[den == 0] = 1  

X_train_norm = (X_train - X_min) / den
X_test_norm = (X_test - X_min) / den

print(f"Train data range: [{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")

class Autoencoder(keras.Model):
    def __init__(self):
        super().__init__()

        self.encoder = keras.Sequential(
            [
                layers.Dense(8, activation="relu", name="encoder_layer1"),
                layers.Dense(5, activation="relu", name="encoder_layer2"),
                layers.Dense(3, activation="relu", name="encoder_layer3"),
            ]
        )

        self.decoder = keras.Sequential(
            [
                layers.Dense(5, activation="relu", name="decoder_layer1"),
                layers.Dense(8, activation="relu", name="decoder_layer2"),
                layers.Dense(9, activation="sigmoid", name="decoder_layer3"),
            ]
        )

    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

autoencoder = Autoencoder()

autoencoder.compile(optimizer="adam", loss="mse")

history = autoencoder.fit(
    X_train_norm,
    X_train_norm,
    epochs=100,
    batch_size=1024,
    validation_data=(X_test_norm, X_test_norm),
    verbose=1,
)

plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Autoencoder Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

X_train_reconstructed = autoencoder.predict(X_train_norm, verbose=0)
train_reconstruction_errors = np.mean(
    np.square(X_train_norm - X_train_reconstructed), axis=1
)

contamination_rate = float(np.mean(y == 1))
print(f"\nContamination rate (dataset): {contamination_rate:.4f}")

threshold = np.quantile(train_reconstruction_errors, 1 - contamination_rate)
print(f"Threshold: {threshold:.6f}")

y_train_pred = (train_reconstruction_errors > threshold).astype(int)
train_balanced_acc = balanced_accuracy_score(y_train, y_train_pred)

X_test_reconstructed = autoencoder.predict(X_test_norm, verbose=0)
test_reconstruction_errors = np.mean(
    np.square(X_test_norm - X_test_reconstructed), axis=1
)

y_test_pred = (test_reconstruction_errors > threshold).astype(int)
test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)

print("\n=== Autoencoder Results ===")
print(f"Train Balanced Accuracy: {train_balanced_acc:.4f}")
print(f"Test Balanced Accuracy:  {test_balanced_acc:.4f}")

print(
    f"\nTrain reconstruction error - Mean: {train_reconstruction_errors.mean():.6f}, "
    f"Std: {train_reconstruction_errors.std():.6f}"
)
print(
    f"Test reconstruction error - Mean: {test_reconstruction_errors.mean():.6f}, "
    f"Std: {test_reconstruction_errors.std():.6f}"
)
print(f"Number of predicted anomalies in train: {int(np.sum(y_train_pred))}")
print(f"Number of predicted anomalies in test:  {int(np.sum(y_test_pred))}")
print(f"Actual anomalies in train: {int(np.sum(y_train))}")
print(f"Actual anomalies in test:  {int(np.sum(y_test))}")
