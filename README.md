# pullup-analyzer

Pull-up analysis tool using Ultralytics YOLO pose estimation and visual feedback.

## What This Project Does

This project reads `.mp4` videos from the `videos/` folder, estimates the athlete's pose with a YOLO pose model, counts pull-up reps, and writes annotated result videos to the `results/` folder.

The visual overlay includes:

- rep count
- grip type
- pull/down phase state
- tempo and cycle metrics
- elbow angle and motion visualization

## Requirements

- Python 3.10+
- `ffmpeg` installed and available in `PATH`
- A supported Ultralytics YOLO pose model

## Package Installation Guide

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Or install the main dependency directly with `pip`:

```bash
pip install ultralytics
```

`ultralytics` installs the core dependencies used by this project, including `numpy` and OpenCV-related packages.

If `ffmpeg` is missing, install it first.

Examples:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y ffmpeg

# macOS (Homebrew)
brew install ffmpeg
```

## Folder Structure

The repository keeps the directory layout in Git, but the folders are empty by default.

```text
pullup-analyzer/
├── demo_pullup_visual.py
├── requirements.txt
├── LICENSE
├── README.md
├── models/
├── videos/
└── results/
```

Folder usage:

- `models/`: place YOLO pose model weights here
- `videos/`: place input `.mp4` files here
- `results/`: generated output videos are written here

## Supported Pose Models

The script is set up to work with these Ultralytics YOLO pose weights:

- `yolo26n-pose.pt`
- `yolo26s-pose.pt`
- `yolo26m-pose.pt`
- `yolo26l-pose.pt`
- `yolo26x-pose.pt`

Official references:

- Ultralytics pose task docs: https://docs.ultralytics.com/tasks/pose/
- `yolo26n-pose.pt`: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt
- `yolo26s-pose.pt`: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt
- `yolo26m-pose.pt`: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-pose.pt
- `yolo26l-pose.pt`: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-pose.pt
- `yolo26x-pose.pt`: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt

## Automatic Model Download

If `models/` does not contain a supported `.pt` file, the script automatically downloads the default model:

```text
models/yolo26m-pose.pt
```

You can request a different supported model by setting `PULLUP_MODEL_NAME` before running the script:

```bash
PULLUP_MODEL_NAME=yolo26s-pose.pt python demo_pullup_visual.py
```

## How To Run

1. Put one or more `.mp4` files into `videos/`.
2. Run the script.
3. Check generated output files in `results/`.

Example:

```bash
python demo_pullup_visual.py
```

## Notes

- Only `.mp4` files in `videos/` are processed.
- Result files are saved as `<original_name>_result.mp4`.
- Original audio is merged back into the output when `ffmpeg` is available.
- The repository does not track large local assets such as videos, results, or model weights.
