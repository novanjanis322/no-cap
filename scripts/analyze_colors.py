"""Analyze color distribution in the dataset."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
import seaborn as sns
def extract_hsv_values(
    image_dir: Path, label_dir: Path
) -> Tuple[List[float], List[float], List[float]]:
    """Extract HSV values from all bottle cap crops.

    Args:
        image_dir: Directory containing images.
        label_dir: Directory containing YOLO labels.

    Returns:
        Tuple of (hue_values, saturation_values, value_values).
    """
    hue_values = []
    sat_values = []
    val_values = []
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"        
        if not label_path.exists():
            continue        
        image = cv2.imread(str(img_path))
        if image is None:
            continue        
        h, w = image.shape[:2]
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue                
                _, x_center, y_center, width, height = map(float, parts)
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue                
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hue_values.append(np.mean(hsv[:, :, 0]))
                sat_values.append(np.mean(hsv[:, :, 1]))
                val_values.append(np.mean(hsv[:, :, 2]))
    return hue_values, sat_values, val_values
def plot_hsv_distribution(
    hue_values: List[float],
    sat_values: List[float],
    val_values: List[float],
    output_path: str = "results/hsv_distribution.png",
) -> None:
    """Plot HSV value distributions.

    Args:
        hue_values: List of hue values.
        sat_values: List of saturation values.
        val_values: List of value (brightness) values.
        output_path: Path to save the plot.
    """    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Hue distribution
    axes[0].hist(hue_values, bins=50, color="red", alpha=0.7)
    axes[0].set_xlabel("Hue (0-179)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Hue Distribution")
    axes[0].axvline(90, color="blue", linestyle="--", label="Blue min (90)")
    axes[0].axvline(130, color="blue", linestyle="--", label="Blue max (130)")
    axes[0].legend()
    # Saturation distribution
    axes[1].hist(sat_values, bins=50, color="green", alpha=0.7)
    axes[1].set_xlabel("Saturation (0-255)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Saturation Distribution")
    axes[1].axvline(50, color="red", linestyle="--", label="Min threshold (50)")
    axes[1].legend()
    # Value distribution
    axes[2].hist(val_values, bins=50, color="blue", alpha=0.7)
    axes[2].set_xlabel("Value/Brightness (0-255)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Value Distribution")
    axes[2].axvline(150, color="red", linestyle="--", label="Light/Dark threshold (150)")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
if __name__ == "__main__":
    # Configure paths    
    image_dir = Path("data/raw/images")
    label_dir = Path("data/raw/labels")
    # Extract HSV values    
    print("Extracting HSV values from dataset...")
    hue_vals, sat_vals, val_vals = extract_hsv_values(image_dir, label_dir)
    print(f"Extracted {len(hue_vals)} samples")
    print(f"Hue range: {min(hue_vals):.1f} - {max(hue_vals):.1f}")
    print(f"Saturation range: {min(sat_vals):.1f} - {max(sat_vals):.1f}")
    print(f"Value range: {min(val_vals):.1f} - {max(val_vals):.1f}")
    # Create results directory    
    Path("results").mkdir(exist_ok=True)
    # Plot distribution    
    plot_hsv_distribution(hue_vals, sat_vals, val_vals)