"""Training pipeline for bottle cap detection."""

from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

import wandb
from bsort.data.preprocessor import DataPreprocessor
from bsort.models.model_factory import ModelFactory


class Trainer:
    """Trainer for bottle cap detection models."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with configuration.

        Args:
            config: Training configuration dictionary.
        """
        self.config = config
        self.model: Optional[YOLO] = None  # Initialize WandB
        if config.get("wandb", {}).get("enabled", False):
            wandb_cfg = config["wandb"]
            wandb.init(
                project=wandb_cfg.get("project", "bottlecap-detection"),
                entity=wandb_cfg.get("entity"),
                name=wandb_cfg.get("name"),
                tags=wandb_cfg.get("tags", []),
                config=config,
            )

    def train(self) -> None:
        """Execute training pipeline."""
        # Step 1: Preprocess data
        print("Step 1: Preprocessing dataset...")
        preprocessor = DataPreprocessor(self.config)
        preprocessor.process_dataset()
        # Step 2: Create model
        print("Step 2: Creating model...")
        self.model = ModelFactory.create_model(self.config["model"])
        # Step 3: Prepare training arguments
        train_cfg = self.config["training"]
        model_cfg = self.config["model"]
        output_cfg = self.config["output"]
        # Create data.yaml for YOLO
        data_yaml_path = self._create_data_yaml()
        # Step 4: Train
        print("Step 3: Starting training...")
        results = self.model.train(
            data=str(data_yaml_path),
            epochs=train_cfg.get("epochs", 100),
            batch=train_cfg.get("batch_size", 16),
            imgsz=model_cfg.get("input_size", 640),
            lr0=train_cfg.get("learning_rate", 0.001),
            optimizer=train_cfg.get("optimizer", "adam"),
            patience=train_cfg.get("early_stopping_patience", 15),
            project=output_cfg.get("model_dir", "models"),
            name="train",
            exist_ok=True,
        )
        # Step 5: Save best model
        print("Step 4: Saving best model...")
        best_model_path = Path(output_cfg["model_dir"]) / output_cfg["best_model_name"]
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        # The best model is saved in runs/detect/train/weights/best.pt
        import shutil

        shutil.copy(
            Path(output_cfg["model_dir"]) / "train" / "weights" / "best.pt",
            best_model_path,
        )
        print(f"Best model saved to {best_model_path}")
        if wandb.run:
            wandb.finish()

    def _create_data_yaml(self) -> Path:
        """Create data.yaml file for YOLO training.

        Returns:
            Path to created data.yaml file.
        """
        # Get absolute paths for train and val directories
        train_images = Path(self.config["data"]["processed_train_images"]).resolve()
        val_images = Path(self.config["data"]["processed_val_images"]).resolve()

        data_yaml_content = f"""train: {train_images}
val: {val_images}
nc: 3
names: ['light_blue', 'dark_blue', 'others']
"""
        yaml_path = Path("data/data.yaml")
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(data_yaml_content)
        print(f"Created {yaml_path}")
        return yaml_path
