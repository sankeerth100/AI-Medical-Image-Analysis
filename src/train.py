import matplotlib
matplotlib.use('Agg')   # 🔥 Fix for Tkinter error (no GUI needed)

import matplotlib.pyplot as plt
from model import build_model
from data_loader import load_data
import os

def train():

    # Create folders if not exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("📂 Loading data...")
    train_data, val_data = load_data()

    print("🧠 Building model...")
    model = build_model()

    print("🚀 Starting training...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=5
    )

    print("💾 Saving model...")
    model.save("models/model.keras")   # ✅ modern format (no warning)

    print("📊 Saving accuracy graph...")
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.savefig("outputs/accuracy.png")

    print("✅ Training Completed Successfully!")

if __name__ == "__main__":
    train()