# pullup-analyzer

Pull-up analysis tool using Ultralytics YOLO pose estimation and visual feedback.

## What This Project Does

This project reads `.mp4` videos from the `videos/` folder, estimates the athlete's pose with a YOLO pose model, counts pull-up reps, and writes annotated result videos to the `results/` folder.

The visual overlay includes:

- rep count
- grip type
- pull/down phase state
- tempo and cycle metrics
- cumulative score and score level
- score trend graph and per-rep score pop-up
- average shoulder rise metric
- best shoulder peak marker
- elbow angle and motion visualization
- compact console summaries with a progress bar for each video

## Requirements

- Python 3.10+
- `av` (PyAV)
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

Or install the main dependencies directly with `pip`:

```bash
pip install ultralytics av tqdm
```

`ultralytics` installs the core dependencies used by this project, including `numpy` and OpenCV-related packages. `av` is used to copy the original audio stream into the rendered output video without relying on an external `ffmpeg` executable.

Important:

- PyAV's pip package name is `av`, not `pyav`
- `pip install pyav` fails because there is no package published under that name

If `pip install av` falls back to a source build on Ubuntu or Debian, install the FFmpeg development libraries first:

```bash
sudo apt update
sudo apt install -y ffmpeg pkg-config \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libavfilter-dev libswscale-dev libswresample-dev
pip install av
```

## Folder Structure

The repository keeps the directory layout in Git, but the folders are empty by default.

```text
pullup-analyzer/
├── demo.py
├── requirements.txt
├── LICENSE
├── README.md
├── pullup_analyzer/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── console.py
│   ├── rendering.py
│   └── state.py
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

The default model is `yolo26s-pose.pt`.

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
models/yolo26s-pose.pt
```

You can request a different supported model by setting `PULLUP_MODEL_NAME` before running the script:

```bash
PULLUP_MODEL_NAME=yolo26s-pose.pt python demo.py
```

## How To Run

1. Put one or more `.mp4` files into `videos/`.
2. Run the script.
3. Check generated output files in `results/`.

Example:

```bash
python demo.py
```

Optional CLI arguments:

```bash
python demo.py --conf 0.50 --iou 0.45
```

## Notes

- Only `.mp4` files in `videos/` are processed.
- Result files are saved as `<original_name>_result.mp4`.
- Original audio is merged back into the output with PyAV when the source video contains audio.
- The repository does not track large local assets such as videos, results, or model weights.
- Inference automatically tries GPU `0` first when CUDA is available, and falls back to CPU if GPU inference is not usable in the current environment.
- Console output shows a color title banner and a fixed two-line live status area so the terminal does not keep scrolling during analysis.
- Scoring is documented in `SCORING.md`. The project now uses a cumulative `Score` model where each rep can earn up to `100` points based on center stability, tempo, height, angle, and a dead-hang bonus.
- Score levels are `초급자 (0~799)`, `중급자 (800~1599)`, `고급자 (1600~2399)`, `마스터 (2400~3999)`, and `신 (4000+)`.
- `Avg Rise` means `Average Shoulder Rise`: the average upward shoulder travel normalized by the athlete's body scale, so it stays comparable across different resolutions and camera distances.
