"""Command-line interface for bsort package."""

from pathlib import Path
from typing import Optional

import click

from bsort.config import load_config
from bsort.inference.predictor import Predictor
from bsort.training.trainer import Trainer


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Bottle cap color detection CLI."""
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to configuration YAML file",
)
def train(config: Path) -> None:
    """Train the bottle cap detection model.

    Args:
        config: Path to the training configuration file.
    """
    click.echo(f"Loading config from {config}")
    cfg = load_config(config)
    trainer = Trainer(cfg)
    trainer.train()
    click.echo("Training Completed.")


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to inference configuration YAML file",
)
@click.option(
    "--image",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the input image for inference",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=False,
    default=None,
    help="Path to save output image (optional)",
)
def infer(config: Path, image: Path, output: Optional[Path]) -> None:
    """Run inference on an image.

    Args:
        config: Path to the inference configuration file.
        image: Path to the input image for inference.
        output: Path to save output image (optional).
    """
    cfg = load_config(config)
    click.echo(f"Running inference on {image}")
    predictor = Predictor(cfg)
    results = predictor.predict(str(image), output_path=str(output) if output else None)
    click.echo(f"Detection completed: {len(results['detections'])} objects detected.")
    click.echo(f"Inference time: {results['inference_time']:.2f}ms")


if __name__ == "__main__":
    cli()
