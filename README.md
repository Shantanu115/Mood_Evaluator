# CoAtNet FER Video Analyzer

A Tkinter-based desktop application for frame-by-frame facial expression recognition using a CoAtNet PyTorch model. This tool analyzes video files to determine emotional trends, tracks negative emotion duration, and calculates the dominant emotion using Global Average Pooling (GAP).

## Dependencies

Ensure you have Python installed with the following libraries:

* torch
* torchvision
* timm
* opencv-python
* matplotlib
* Pillow
* numpy
* tk

## Installation

1.  Clone the repository or download the script.
2.  Install the required packages:
    ```bash
    pip install torch torchvision timm opencv-python matplotlib Pillow numpy
    ```
3.  Ensure you have a trained PyTorch model file (`.pt` or `.pth`) compatible with `coatnet_0_rw_224` architecture.

## Usage

1.  Run the application:
    ```bash
    python "FER video GUI.py"
    ```
2.  **Load Model**: Click "1. Load Model" and select your trained weights file.
3.  **Load Video**: Click "2. Load Video" to select an `.mp4` or `.avi` file.
4.  **Start Analysis**: Click "START" to begin inference.

## Features

* **Real-time Inference**: Detects faces and classifies emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) frame-by-frame.
* **Trend Graph**: Plots emotional valence (Positive, Ambiguous, Negative) over time using Matplotlib.
* **Negative Log**: Automatically logs the start time and duration of sustained negative emotions.
* **GAP Analysis**: Calculates the "Dominant Emotion" for the entire session upon completion.

## Model Mapping

The application groups 7 discrete emotions into 3 categories for graph visualization:

* **Positive**: Happy
* **Ambiguous**: Neutral, Surprise
* **Negative**: Anger, Disgust, Fear, Sad
