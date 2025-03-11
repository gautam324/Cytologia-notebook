

# YOLOv8 Training for CytologIA Dataset

## Overview

This repository contains code for training YOLOv8-S on the CytologIA dataset, which involves classifying and detecting leukocytes into 23 distinct classes. The dataset is part of the CytologIA Data Challenge, hosted by Trustii.io, the French Cellular Hematology Group, and Algoscope, supported by the France 2030 plan and the Health Data Hub.

https://app.trustii.io/leaderboard-v2/1533

## Features

- **YOLOv8-S Model**: Utilized for object detection.
- **Automated Training Pipeline**: Custom training function for streamlined execution.
- **Dataset Preprocessing**: Ensures proper structure and formatting.
- **Validation and Testing**: Performance evaluation using appropriate metrics.

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed and set up a virtual environment:

```bash
python -m venv yolov8_env
source yolov8_env/bin/activate  # On Windows use yolov8_env\Scripts\activate
```

### Install Dependencies

```bash
pip install ultralytics opencv-python numpy torch torchvision tqdm
```

## Training the Model

The training function `train_yolov8()` is structured as follows:

```python
from ultralytics import YOLO

def train_yolov8(data_yaml, epochs=50, batch_size=16, img_size=640):
    model = YOLO("yolov8s.pt")  # Load YOLOv8-S model
    model.train(data=data_yaml, epochs=epochs, batch=batch_size, imgsz=img_size)
```

To train the model, run:

```python
train_yolov8("path/to/data.yaml", epochs=100, batch_size=32)
```

## Dataset Structure

Ensure your dataset follows the YOLO format:

```
/dataset
    /images
        /train
        /val
    /labels
        /train
        /val
    data.yaml
```

## Evaluation

After training, evaluate the model using:

```python
model.val()
```

## Inference

Run inference on an image:

```python
results = model("test_image.jpg")
results.show()
```

## Performance Metrics

- **Dataset**: CytologIA Data Challenge
- **Cytologia Metric**: `0.8284`

## Acknowledgments

- **CytologIA Data Challenge**
- **Trustii.io, Algoscope, France 2030 Plan, and Health Data Hub**
- **Ultralytics YOLOv8**

## License

This project is licensed under the MIT License.

