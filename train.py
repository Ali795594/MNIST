import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
import os
from config import Config

def train_model():
    print("🚀 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    print("🏗️ Building CNN Model...")
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("📚 Training model... (Takes 30-90 seconds)")
    model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    os.makedirs('model', exist_ok=True)
    model.save(Config.MODEL_PATH)
    print(f"✅ Model saved at: {Config.MODEL_PATH}")
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n🎉 Training Completed! Test Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    train_model()