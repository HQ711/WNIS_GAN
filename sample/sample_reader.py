#!/usr/bin/env python3
"""
Minimal example for reading the P1 interview package data.

This script intentionally does not implement training logic. It only shows:
1. how to read one `.npz` file,
2. what keys are available,
3. the basic array shapes.
"""

from pathlib import Path

import numpy as np


def main():
    base_dir = Path(__file__).resolve().parents[1]
    example_path = base_dir / "data" / "val" / "p1_4_spherical_grid.npz"
    if not example_path.exists():
        raise FileNotFoundError(f"Example file not found: {example_path}")

    data = np.load(example_path)

    print("=" * 72)
    print("Sample NPZ Reader")
    print("=" * 72)
    print(f"File: {example_path.name}")
    print()
    print("Available keys:")
    for key in data.files:
        print(f"  - {key}")

    print()
    print("Shapes:")
    for key in data.files:
        value = data[key]
        print(f"  {key:22s} shape={value.shape} dtype={value.dtype}")

    if "velocity" in data.files:
        velocity = data["velocity"]
        print()
        print(f"Velocity tensor shape: {velocity.shape}")
        print(f"Frame count: {velocity.shape[0]}")
        print("This example file is one full anonymized P1 training recording.")

    if "frame_indices" in data.files:
        frame_indices = data["frame_indices"]
        print(f"First frame index: {frame_indices[0]}")
        print(f"Last frame index : {frame_indices[-1]}")


if __name__ == "__main__":
    main()
