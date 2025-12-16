import os
from dataset import load_dataset, split_dataset, build_augmentor
from model import build_model
from utils import plot_and_save_history
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def main():
    # Resolve paths relative to this file so training works from any CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.dirname(script_dir)

    data_dir = os.path.join(test_dir, "data")
    results_dir = os.path.join(test_dir, "results")
    saved_model_dir = os.path.join(test_dir, "saved_model")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)

    X, y = load_dataset(
        with_mask_path=os.path.join(data_dir, "with_mask"),
        without_mask_path=os.path.join(data_dir, "without_mask")
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    datagen = build_augmentor()
    datagen.fit(X_train)

    model = build_model()

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(saved_model_dir, "best_model.h5"),
        save_best_only=True,
        monitor="val_loss"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=25,
        callbacks=[checkpoint, early_stop]
    )

    plot_and_save_history(history, results_dir)

    print("Training complete.")

if __name__ == "__main__":
    main()
