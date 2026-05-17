# train_script.py — YOLOv8 Training Script for CS2 Player Detection
# Trains a custom YOLOv8 model to detect CT and T players in Counter-Strike 2

import os
import argparse
import shutil
# pyrefly: ignore [missing-import]
from ultralytics import YOLO


def train(
    data_yaml: str = os.path.join("dataset", "data.yaml"),
    model_name: str = os.path.join("models", "best.pt"),
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/train",
    name: str = "cs2_player_detection",
    patience: int = 50,
    save_period: int = 5,
):
    """
    Train a YOLOv8 model for CS2 player detection.

    Args:
        data_yaml: Path to the dataset YAML configuration file
        model_name: YOLOv8 pretrained model to use (n/s/m/l/x variants)
        epochs: Number of training epochs
        imgsz: Input image size for the model
        batch: Batch size for training
        device: Device to train on ('gpu', '0', '0,1,2,3', etc.)
        project: Project directory to save training results
        name: Name of the training run
        patience: Early stopping patience (epochs without improvement)
        save_period: Save checkpoint every N epochs
    """
    # Load pretrained YOLOv8 model
    print(f"Loading pretrained model: {model_name}")
    model = YOLO(model_name)

    # Start training
    print(f"Starting training with dataset: {data_yaml}")
    print(f"Device: {device}, Batch: {batch}, Image size: {imgsz}")
    print(f"Epochs: {epochs}, Patience: {patience}")
    print("-" * 50)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save_period=save_period,
        # Training hyperparameters (tuned for object detection)
        lr0=0.005,        # Lower initial LR for fine-tuning
        lrf=0.01,         # Final learning rate
        momentum=0.937,     # SGD momentum/Adam beta
        weight_decay=0.0005, # Optimizer weight decay
        warmup_epochs=3.0,   # Warmup epochs
        warmup_momentum=0.8, # Warmup momentum
        cls=1.0,          # Higher class loss weight (was 0.5)
        box=10.0,         # Higher box loss weight (was 7.5)
        dfl=1.5,          # DFL loss gain
        # Augmentation
        hsv_h=0.015,        # HSV-Hue augmentation
        hsv_s=0.7,          # HSV-Saturation augmentation
        hsv_v=0.4,          # HSV-Value augmentation
        degrees=10.0,        # Rotation degrees
        translate=0.1,      # Translation augmentation
        scale=0.3,          # Scaling augmentation
        shear=0.0,          # Shear augmentation
        perspective=0.0,    # Perspective augmentation
        flipud=0.0,         # Vertical flip probability
        fliplr=0.5,         # Horizontal flip probability
        mosaic=1.0,         # Mosaic augmentation probability
        mixup=0.1,          # Mixup augmentation probability
        cutmix=0.1,         
        # Other
        verbose=True,
        exist_ok=True,
    )

    print("-" * 50)
    print("Training completed!")
    print(f"Results saved to: {os.path.join(project, name)}")
    print(f"Best model: {os.path.join(project, name, 'weights', 'best.pt')}")
    print(f"Last model: {os.path.join(project, name, 'weights', 'last.pt')}")

    # Copy the best model to the models/ directory
    best_model_path = os.path.join(project, name, 'weights', 'best.pt')
    dest_model_path = os.path.join("models", "best.pt")
    os.makedirs("models", exist_ok=True)
    shutil.copy(best_model_path, dest_model_path)
    print(f"Copied best model to: {dest_model_path}")

    return results


def validate(
    model_path: str,
    data_yaml: str = os.path.join("dataset", "data.yaml"),
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
):
    """
    Validate a trained YOLOv8 model.

    Args:
        model_path: Path to the trained model weights
        data_yaml: Path to the dataset YAML configuration file
        imgsz: Input image size for the model
        batch: Batch size for validation
        device: Device to validate on
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Starting validation with dataset: {data_yaml}")
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
    )

    print("-" * 50)
    print("Validation completed!")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for CS2 player detection")

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Mode: 'train' or 'val'",
    )

    # Training arguments
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("dataset", "data.yaml"),
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("models", "best.pt"),
        help="Path to model weights for fine-tuning (e.g., models/best.pt) or a YOLOv8 variant",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use (0, 0,1,2,3, cpu)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="cs2_player_detection",
        help="Experiment name",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience",
    )

    # Validation arguments
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained model weights (for validation mode)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
        )
    elif args.mode == "val":
        if not args.weights:
            print("Error: --weights is required for validation mode")
            exit(1)
        validate(
            model_path=args.weights,
            data_yaml=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
