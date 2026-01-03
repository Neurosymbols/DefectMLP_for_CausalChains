# run.py
import argparse
import torch

from app.data_prep import prepare_data_for_training, save_preprocessing_artifacts
from app.dataset_loader import (
    create_dataloaders,
    compute_class_weights,
    compute_mechanism_weights
)
from app.model import create_model
from app.train import train_model, TrainingConfig
from app.evaluate.evaluate_multitask import evaluate_multitask_simple
from app.inference import predict_board
from app.model_loading import load_model


PREPROCESSING_PATH = "./app/multitask_model/preprocessing_artifacts.pkl"
MODEL_PATH = "./app/multitask_model/best_multitask_model.pth"


def run_data_prep():
    print("Preparing data...")
    data = prepare_data_for_training(data_source="mongodb")

    save_preprocessing_artifacts(
        data["label_encoder"],
        data["scaler"],
        data["feature_info"],
        PREPROCESSING_PATH
    )

    print("Creating dataloaders...")
    dataloaders = create_dataloaders(
        data["X_train"], data["y_train"], data["y_mechanism_train"],
        data["X_val"], data["y_val"], data["y_mechanism_val"],
        data["X_test"], data["y_test"], data["y_mechanism_test"],
        batch_size=256,
        num_workers=0
    )

    return data, dataloaders


def run_training(data, dataloaders):
    print("\nComputing weights...")
    defect_weights = compute_class_weights(data["y_train"])
    mechanism_weights = compute_mechanism_weights(data["y_mechanism_train"])

    print("\nCreating model...")
    model = create_model("multitask", input_dim=14, num_classes=3)
    print(model.get_architecture_summary())

    config = TrainingConfig()

    task_weights = {
        "defect": 1.0,
        "mechanism": 0.5
    }

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    model, history = train_model(
        model,
        dataloaders,
        defect_weights,
        mechanism_weights,
        config,
        task_weights,
        save_path=MODEL_PATH
    )

    print("\n✓ Training complete!")
    print(f"Val Defect F1:    {history['val_defect_f1'][-1]:.4f}")
    print(f"Val Mechanism F1: {history['val_mechanism_f1'][-1]:.4f}")


def run_inference(dataloaders):
    print("\nLoading model for inference...")
    model, scaler, features = load_model(
        model_path=MODEL_PATH,
        preprocessing_path=PREPROCESSING_PATH
    )

    print("\nRunning evaluation...")
    evaluate_multitask_simple(model, dataloaders["test"])

    print("\nRunning single-board inference...")
    board = {
        "paste_volume": 0.035,
        "stencil_thickness": 96,
        "paste_viscosity": 257,
        "ambient_rh": 40,
        "ambient_temperature": 25
    }

    result = predict_board(board, model, scaler, features)
    print(result)


def main():
    parser = argparse.ArgumentParser(description="Multitask SMT Model Pipeline")

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training stage"
    )

    parser.add_argument(
        "--infer",
        action="store_true",
        help="Run evaluation and inference stage"
    )

    args = parser.parse_args()

    # ALWAYS RUN DATA PREP
    data, dataloaders = run_data_prep()

    if args.train:
        run_training(data, dataloaders)

    if args.infer:
        run_inference(dataloaders)

    if not args.train and not args.infer:
        print("\nOnly data preparation was run. Use --train or --infer.")


if __name__ == "__main__":
    main()
