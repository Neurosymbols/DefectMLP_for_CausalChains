# run.py
import argparse
import torch

from app.data_prep import prepare_data_for_training, save_preprocessing_artifacts
from app.dataset_loader import (
    create_dataloaders,
    compute_class_weights
)
from app.model import create_model
from app.train import train_model, TrainingConfig
from app.evaluate.evaluate_multitask import evaluate_multitask_simple
from app.mlp_inference import predict, prepare_features, load_model


PREPROCESSING_PATH = "./app/multitask_model/preprocessing_artifacts.pkl"
MODEL_PATH = "./app/multitask_model/best_multitask_model.pth"


def run_data_prep():
    print("Preparing data...")
    data = prepare_data_for_training(data_source='mongodb')
    save_preprocessing_artifacts(
        data['defect_encoder'],
        data['print_encoder'],
        data['reflow_encoder'],
        data['scaler'],
        data['feature_info'],
        "./app/multitask_model/preprocessing_artifacts.pkl"
    )
    # Create dataloaders
    dataloaders = create_dataloaders(
        data,
        batch_size=256,
        num_workers=0
    )

    return data, dataloaders


def run_training(data, dataloaders):
    print("\nComputing weights...")
    # Compute weights
    defect_weights = compute_class_weights(data['y_train'])
    print_mechanism_weights = compute_class_weights(data['y_print_mech_train'])
    reflow_mechanism_weights = compute_class_weights(data['y_reflow_mech_train'])
    # Per-parameter weights
    param_weights = torch.tensor([
        1.0,  # paste_volume (learning well)
        1.5,  # stencil_thickness (learning well)
        4.0,  # paste_viscosity
        5.0,  # ambient_rh
        3.0,  # ambient_temperature
        5.0,  # peak_reflow_temperature
        5.0,  # time_above_liquidus
    ])

    print("\nCreating model...")
    # Create multi-task model
    model = create_model(
        'full_multitask_multistage', 
        input_dim = len(data['feature_info']['all_features']), 
        num_defect_classes = 3, 
        num_mechanism_stages_classes={'print': 3, 'reflow': 3},
        num_parameters = len(data['feature_info']['raw'])
    )

    # Print summary
    print(model.get_architecture_summary())

    config = TrainingConfig()

    config.epochs = 80

    task_weights = {
        "defect": 0.8,
        "print_mechanism": 0.8,
        "reflow_mechanism": 1.0, # since occurances are less
        'param_risk': 1.5
    }

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    model, history = train_model(
        model,
        dataloaders,
        defect_weights,
        print_mechanism_weights,
        reflow_mechanism_weights,
        config,
        task_weights,
        param_weights,
        save_path=MODEL_PATH
    )

    print("\n✓ Training complete!")
    print(f"Val Defect F1:    {history['val_defect_f1'][-1]:.4f}")
    print(f"Val Printing Mechanism F1: {history['val_print_mechanism_f1'][-1]:.4f}")
    print(f"Val Reflow Mechanism F1: {history['val_reflow_mechanism_f1'][-1]:.4f}")
    print(f"MAE Param Viol: {history['val_param_risk_mae'][-1]:.4f}")


def run_inference(dataloaders):
    print("\nLoading model for inference...")
    model, scaler, features = load_model(
        model_path=MODEL_PATH,
        preprocessing_path=PREPROCESSING_PATH
    )

    # print("\nRunning evaluation...")
    # evaluate_multitask_simple(model, dataloaders["test"])

    print("\nRunning single-board inference...")
    board = {
        "paste_volume": 0.035,
        "stencil_thickness": 96,
        "paste_viscosity": 257,
        "ambient_rh": 40,
        "ambient_temperature": 25
    }

    features = prepare_features(board, scaler, features)
    result = predict(
        model,
        features
    )
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
