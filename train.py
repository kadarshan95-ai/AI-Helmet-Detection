from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the helmet detector using a YOLO-format dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=BASE_DIR / "dataset.yaml",
        help="Dataset YAML path (default: dataset.yaml beside this script).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=BASE_DIR / "yolov8n.pt",
        help="Pretrained YOLO weights used to start training.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, or a CUDA device such as 0.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", default="helmet_detection")
    return parser.parse_args()


def validate_dataset(data_file: Path) -> None:
    if not data_file.is_file():
        raise SystemExit(
            f"Dataset YAML not found: {data_file}\n"
            "Create dataset.yaml or pass its location with --data."
        )

    try:
        config: Any = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise SystemExit(f"Could not parse dataset YAML {data_file}: {error}") from error

    if not isinstance(config, dict):
        raise SystemExit(f"Dataset YAML must contain a mapping: {data_file}")

    configured_root = Path(config.get("path") or ".")
    dataset_root = (
        configured_root
        if configured_root.is_absolute()
        else (data_file.parent / configured_root).resolve()
    )
    missing: list[Path] = []
    empty: list[Path] = []

    for split in ("train", "val"):
        split_value = config.get(split)
        if not split_value:
            raise SystemExit(
                f"Dataset YAML is missing the required `{split}:` entry: {data_file}"
            )
        split_paths = split_value if isinstance(split_value, list) else [split_value]
        for split_path in split_paths:
            if not isinstance(split_path, str):
                raise SystemExit(
                    f"Dataset `{split}` entries must be paths in {data_file}."
                )
            image_dir = Path(split_path)
            if not image_dir.is_absolute():
                image_dir = dataset_root / image_dir
            image_dir = image_dir.resolve()
            if not image_dir.is_dir():
                missing.append(image_dir)
            elif not any(
                path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                for path in image_dir.rglob("*")
            ):
                empty.append(image_dir)

    if missing or empty:
        if missing:
            print("Dataset image folders are missing:")
            for path in missing:
                print(f"  - {path}")
        if empty:
            print("Dataset image folders contain no supported images:")
            for path in empty:
                print(f"  - {path}")
        raise SystemExit(
            "\nAdd labeled images to the train and validation locations in "
            f"{data_file}, or point to another dataset with --data."
        )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1 or args.workers < 0:
        raise SystemExit(
            "epochs, batch, and imgsz must be positive (imgsz at least 32); "
            "workers cannot be negative."
        )

    data_file = args.data.expanduser().resolve()
    validate_dataset(data_file)

    if not args.model.is_file():
        raise SystemExit(f"Initial model weights not found: {args.model}")

    device = None if args.device.casefold() == "auto" else args.device
    model = YOLO(str(args.model.resolve()))
    print(f"Starting training with dataset {data_file}")
    model.train(
        data=str(data_file),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=str(BASE_DIR / "runs" / "detect"),
        name=args.name,
    )

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise RuntimeError(f"Training finished but best weights were not created: {best_weights}")

    print(f"Training completed. Best weights: {best_weights}")
    print("Validating the best model...")
    metrics = model.val(data=str(data_file))
    print(metrics)


if __name__ == "__main__":
    main()
