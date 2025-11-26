# Bottle Cap Color Detection

A computer vision system for detecting and classifying bottle caps by color using YOLOv8. The system distinguishes between light blue, dark blue, and other colored bottle caps.

## Overview

This project implements an ML pipeline for bottle cap detection with automatic color-based re-labeling using HSV color space analysis. The dataset is re-labeled from the original annotations to classify caps into three categories based on their color.

**Current Status:**
- Model training and inference pipeline: ✅ Implemented
- CLI interface: ✅ Working
- WandB experiment tracking: ✅ Integrated
- CI/CD pipeline: ❌ Not implemented
- Performance target: ❌ Not met (see [Results](#results) section)

## Project Structure

```
ada-mata-test/
├── bsort/                       # Main package
│   ├── cli.py                   # Command-line interface
│   ├── config.py                # Configuration management
│   ├── data/
│   │   └── preprocessor.py      # Dataset preprocessing and color-based labeling
│   ├── models/
│   │   └── model_factory.py     # Model creation
│   ├── training/
│   │   └── trainer.py           # Training pipeline
│   ├── inference/
│   │   ├── predictor.py         # Inference engine
│   │   └── visualizer.py        # Result visualization
│   └── utils/
│       └── color_classifier.py  # HSV color classification
├── configs/
│   ├── training_config.yaml     # Training configuration
│   └── infer_config.yaml        # Inference configuration
├── tests/                       # Unit tests
├── scripts/
│   └── analyze_colors.py        # Color distribution analysis
└── pyproject.toml               # Dependencies
```

## Installation

**Prerequisites:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/novanjanis322/no-cap.git
cd no-cap

# Create virtual environment and install dependencies (using uv)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Verify installation
bsort --help
```

## Dataset Setup

1. Download the dataset

2. Extract to `data/raw/`:
   ```
   data/raw/
   ├── images/
   └── labels/
   ```

3. (Optional) Analyze color distribution:
   ```bash
   python scripts/analyze_colors.py
   ```

## Usage

### Training

```bash
bsort train --config configs/training_config.yaml
```

This will:
- Re-label the dataset based on HSV color analysis
- Split data into train/val sets (80/20)
- Train YOLOv8n model
- Save best model to `models/best_model.pt`
- Log metrics to WandB

### Inference

```bash
bsort infer --config configs/infer_config.yaml --image path/to/image.jpg --output results/output.jpg
```

## Results

### Model Performance

**Validation Metrics** (from training):

| Metric | Overall | Dark Blue | Others |
|--------|---------|-----------|--------|
| Precision | 0.846 | 0.692 | 1.000 |
| Recall | 0.851 | 1.000 | 0.702 |
| mAP@0.5 | 0.972 | 0.995 | 0.949 |
| mAP@0.5:0.95 | 0.933 | 0.970 | 0.896 |

*Note: Light blue class had insufficient validation samples in this run.*

### Inference Speed

**Test Environment:**
- Hardware: AMD Ryzen 7 5800H
- Model: YOLOv8n (3M parameters)
- Input size: 640×640

**Results:**

| Metric | Value |
|--------|-------|
| Inference time | ~110ms |
| Target (Raspberry Pi 5) | 5-10ms |
| **Gap** | **11-22x slower** |

⚠️ **Performance Issue:** Current implementation does not meet the required inference time target. The model needs significant optimization for edge device deployment.

## Model Details

### Architecture
- **Model:** YOLOv8n (Ultralytics)
- **Parameters:** 3.0M
- **Input:** 640×640
- **Classes:** 3 (light_blue, dark_blue, others)

### Color Classification (HSV)

The system re-labels bounding boxes based on HSV color analysis:

- **Light Blue:** H: 90-130, S: ≥50, V: ≥150
- **Dark Blue:** H: 90-130, S: ≥50, V: <150
- **Others:** Everything else

### Training Configuration
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 16
- Epochs: 100
- Early stopping: 15 epochs patience

## WandB Integration

The training pipeline integrates with [Weights & Biases](https://wandb.ai) for experiment tracking.

Setup:
```bash
wandb login
```

Configure in `configs/training_config.yaml`:
```yaml
wandb:
  enabled: true
  project: "bottlecap-detection"
  entity: ""  # Set your WandB username
```

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
# Format code
black bsort/ tests/ scripts/
isort bsort/ tests/ scripts/

# Lint
pylint bsort/ --fail-under=7.0
```

## Known Issues & Limitations

1. **Inference Speed:** Does not meet the 5-10ms target for Raspberry Pi 5 deployment
2. **CI/CD:** GitHub Actions pipeline not implemented
3. **Light Blue Class:** Limited validation samples in current dataset split
4. **Optimization:** No model quantization or pruning applied yet

## Future Work

- Model optimization (quantization, pruning, distillation)
- TensorRT/ONNX conversion for faster inference
- Raspberry Pi 5 deployment and benchmarking
- CI/CD pipeline implementation (black, isort, pylint, pytest, Docker build)
- Expanded test coverage


**Last Updated:** November 2025
