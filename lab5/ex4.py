import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print(f"Train data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

noise_factor = 0.35
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0.0, clip_value_max=1.0)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0.0, clip_value_max=1.0)

print(f"Noisy train data range: [{x_train_noisy.numpy().min():.4f}, {x_train_noisy.numpy().max():.4f}]")

class ConvAutoencoder(keras.Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        self.encoder = keras.Sequential([
            layers.Conv2D(8, (3, 3), activation='relu', strides=2, padding='same', name='encoder_conv1'),
            layers.Conv2D(4, (3, 3), activation='relu', strides=2, padding='same', name='encoder_conv2')
        ])
        
        self.decoder = keras.Sequential([
            layers.Conv2DTranspose(4, (3, 3), activation='relu', strides=2, padding='same', name='decoder_conv1'),
            layers.Conv2DTranspose(8, (3, 3), activation='relu', strides=2, padding='same', name='decoder_conv2'),
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoder_output')
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = ConvAutoencoder()

autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, x_test),
    verbose=1
)

x_train_reconstructed = autoencoder.predict(x_train)
train_reconstruction_errors = np.mean(np.square(x_train - x_train_reconstructed), axis=(1, 2, 3))

threshold = np.mean(train_reconstruction_errors) + np.std(train_reconstruction_errors)
print(f"\nThreshold: {threshold:.6f}")
print(f"Mean reconstruction error: {np.mean(train_reconstruction_errors):.6f}")
print(f"Std reconstruction error: {np.std(train_reconstruction_errors):.6f}")

x_test_reconstructed = autoencoder.predict(x_test)
test_reconstruction_errors = np.mean(np.square(x_test - x_test_reconstructed), axis=(1, 2, 3))
y_test_pred = (test_reconstruction_errors > threshold).astype(int)

x_test_noisy_reconstructed = autoencoder.predict(x_test_noisy)
test_noisy_reconstruction_errors = np.mean(np.square(x_test_noisy - x_test_noisy_reconstructed), axis=(1, 2, 3))
y_test_noisy_pred = (test_noisy_reconstruction_errors > threshold).astype(int)

accuracy_original = 1 - np.mean(y_test_pred)  
accuracy_noisy = np.mean(y_test_noisy_pred)   

print(f"\n=== Classification Results ===")
print(f"Accuracy on original test images (should be normal): {accuracy_original:.4f}")
print(f"Accuracy on noisy test images (should be anomalies): {accuracy_noisy:.4f}")
print(f"Original images classified as normal: {np.sum(y_test_pred == 0)} / {len(y_test_pred)}")
print(f"Noisy images classified as anomalies: {np.sum(y_test_noisy_pred == 1)} / {len(y_test_noisy_pred)}")

n_images = 5
indices = np.random.choice(len(x_test), n_images, replace=False)

fig, axes = plt.subplots(4, n_images, figsize=(15, 12))

for i, idx in enumerate(indices):
    axes[0, i].imshow(x_test[idx].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Original', fontsize=10)
    
    axes[1, i].imshow(x_test_noisy[idx].numpy().squeeze(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Noisy', fontsize=10)
    
    axes[2, i].imshow(x_test_reconstructed[idx].squeeze(), cmap='gray')
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_title('Reconstructed\n(from original)', fontsize=10)
    
    axes[3, i].imshow(x_test_noisy_reconstructed[idx].squeeze(), cmap='gray')
    axes[3, i].axis('off')
    if i == 0:
        axes[3, i].set_title('Reconstructed\n(from noisy)', fontsize=10)

plt.suptitle('Autoencoder Results', fontsize=14, y=0.98)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Training Denoising Autoencoder...")
print("="*60)

denoising_autoencoder = ConvAutoencoder()
denoising_autoencoder.compile(optimizer='adam', loss='mse')

history_denoising = denoising_autoencoder.fit(
    x_train_noisy, x_train,  
    epochs=10,
    batch_size=64,
    validation_data=(x_test_noisy, x_test),
    verbose=1
)

x_test_reconstructed_denoising = denoising_autoencoder.predict(x_test)
x_test_noisy_reconstructed_denoising = denoising_autoencoder.predict(x_test_noisy)

fig, axes = plt.subplots(4, n_images, figsize=(15, 12))

for i, idx in enumerate(indices):
    axes[0, i].imshow(x_test[idx].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Original', fontsize=10)
    
    axes[1, i].imshow(x_test_noisy[idx].numpy().squeeze(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Noisy', fontsize=10)
    
    axes[2, i].imshow(x_test_reconstructed_denoising[idx].squeeze(), cmap='gray')
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_title('Reconstructed\n(from original)', fontsize=10)
    
    axes[3, i].imshow(x_test_noisy_reconstructed_denoising[idx].squeeze(), cmap='gray')
    axes[3, i].axis('off')
    if i == 0:
        axes[3, i].set_title('Reconstructed\n(from noisy)', fontsize=10)

plt.suptitle('Denoising Autoencoder Results', fontsize=14, y=0.98)
plt.tight_layout()
plt.show()

print("\n=== Comparison: Regular vs Denoising Autoencoder ===")
mse_regular_clean = np.mean(np.square(x_test[:100] - x_test_reconstructed[:100]))
mse_regular_noisy = np.mean(np.square(x_test[:100] - x_test_noisy_reconstructed[:100]))
mse_denoising_clean = np.mean(np.square(x_test[:100] - x_test_reconstructed_denoising[:100]))
mse_denoising_noisy = np.mean(np.square(x_test[:100] - x_test_noisy_reconstructed_denoising[:100]))

print(f"Regular AE - MSE on clean images: {mse_regular_clean:.6f}")
print(f"Regular AE - MSE on noisy images: {mse_regular_noisy:.6f}")
print(f"Denoising AE - MSE on clean images: {mse_denoising_clean:.6f}")
print(f"Denoising AE - MSE on noisy images (denoised): {mse_denoising_noisy:.6f}")