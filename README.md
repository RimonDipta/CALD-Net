# CALD-Net

## Condition-Adaptive Lightweight Detector

CALD-Net is a research-oriented lightweight object detection project focused on building an efficient detector that can remain reliable under challenging visual conditions while requiring relatively modest computational resources.

The project investigates lightweight detection under conditions such as:

- Low light
- Extreme darkness
- Overexposure
- Saturation
- Backlighting
- Low contrast
- Image noise
- Motion blur
- Defocus blur

The architecture is being developed incrementally. Each major component is introduced, tested, and evaluated separately so that architectural changes can be measured against a reproducible baseline.

---

## Research Goal

The main research goal is to investigate whether a lightweight object detector can achieve a useful balance between:

1. Detection accuracy
2. Computational efficiency
3. Robustness to difficult visual conditions
4. Low hardware requirements
5. Practical inference speed

The current implementation is the initial baseline and is **not yet the final CALD-Net architecture**.

---

## Current Status

### CALD-Net v0.1 — Baseline Detection Pipeline

The current baseline contains:

- Lightweight CNN backbone
- Multi-scale feature extraction
- Feature Pyramid Network (FPN)
- Multi-scale detection head
- Bounding-box regression
- Objectness prediction
- Multi-class classification
- Target assignment
- Detection loss
- Bounding-box decoding
- Confidence filtering
- Class-aware Non-Maximum Suppression (NMS)
- Precision evaluation
- Recall evaluation
- Average Precision (AP)
- mAP@50
- mAP@50:95

The current pipeline has been tested using synthetic detection data.

Real-dataset benchmarking has not yet been performed.

---

## Current Architecture

The initial baseline follows:

```text
                    Input Image
                         │
                         ▼
              Lightweight CNN Backbone
                         │
                 ┌───────┼───────┐
                 ▼       ▼       ▼
                P3      P4      P5
                 │       │       │
                 └───────┼───────┘
                         ▼
                        FPN
                         │
                 ┌───────┼───────┐
                 ▼       ▼       ▼
                F3      F4      F5
                 │       │       │
                 └───────┼───────┘
                         ▼
               Multi-Scale Detection Head
                         │
                         ▼
                  Raw Predictions
                         │
                         ▼
                       Decode
                         │
                         ▼
                   Confidence Filter
                         │
                         ▼
                        NMS
                         │
                         ▼
                 Final Detections
```

For a `640 × 640` input, the baseline produces three feature levels:

```text
P3 → 80 × 80
P4 → 40 × 40
P5 → 20 × 20
```

with effective strides:

```text
P3 → stride 8
P4 → stride 16
P5 → stride 32
```

---

## Repository Structure

```text
CALD-Net/
│
├── caldnet/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic.py
│   │   └── collate.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── blocks.py
│   │   ├── backbone.py
│   │   ├── neck.py
│   │   ├── head.py
│   │   ├── detector.py
│   │   ├── box_ops.py
│   │   ├── assigner.py
│   │   ├── targets.py
│   │   ├── loss.py
│   │   ├── decode.py
│   │   ├── inference.py
│   │   ├── nms.py
│   │   └── detector_inference.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── detection.py
│       ├── metrics.py
│       └── map.py
│
├── scripts/
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_iou.py
│   ├── test_targets.py
│   ├── test_loss.py
│   ├── train_synthetic.py
│   ├── test_dataset.py
│   ├── test_dataloader.py
│   ├── train_dataloader.py
│   ├── test_decode.py
│   ├── test_inference.py
│   ├── test_nms.py
│   ├── test_detector_inference.py
│   ├── test_evaluation.py
│   ├── test_map.py
│   └── test_map_50_95.py
│
├── notes/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Baseline Components

### Backbone

The baseline backbone is a simple lightweight CNN consisting of convolutional blocks with:

```text
Conv2d
   ↓
BatchNorm
   ↓
SiLU
```

It progressively reduces spatial resolution while increasing feature channels.

Current channel progression:

```text
Input
  ↓
3 → 32
  ↓
32 → 64
  ↓
64 → 128
  ↓
128 → 256
  ↓
256 → 512
```

### Feature Pyramid Network

The baseline uses a lightweight top-down FPN.

The deepest feature is reduced and upsampled, then fused with the preceding feature level.

This produces three feature maps with a common channel width of 128 channels.

### Detection Head

Each feature level predicts:

```text
4 box values
1 objectness value
N class values
```

Therefore the output channel count is:

```text
5 + number_of_classes
```

The three prediction levels correspond to P3, P4, and P5.

### Target Assignment

The current baseline uses a simple center-based assignment strategy.

Ground-truth boxes use:

```text
[x_center, y_center, width, height, class_id]
```

with normalized coordinates.

Objects are assigned to feature levels based on their approximate size.

Current thresholds:

```text
small  → P3
medium → P4
large  → P5
```

This is an educational baseline and is expected to be improved later.

### Loss

The current detection loss consists of:

```text
L_total =
    L_box
    + L_objectness
    + L_classification
```

The baseline uses:

- Smooth L1 for box regression
- BCE with logits for objectness
- BCE with logits for classification

The current loss implementation is intentionally simple and will likely be redesigned as the detector becomes more sophisticated.

---

## Inference Pipeline

The inference process is:

```text
Raw model output
       ↓
Decode boxes
       ↓
Sigmoid objectness
       ↓
Sigmoid class probabilities
       ↓
Objectness × class probability
       ↓
Confidence threshold
       ↓
Class-wise NMS
       ↓
Final detections
```

---

## Evaluation

CALD-Net currently includes an evaluation pipeline based on IoU matching.

### Intersection over Union

For two bounding boxes:

```text
IoU = Intersection Area / Union Area
```

A prediction is considered a match when:

1. The predicted class matches the ground-truth class.
2. The IoU reaches the selected threshold.
3. The ground-truth object has not already been matched.

### Precision

```text
Precision = TP / (TP + FP)
```

### Recall

```text
Recall = TP / (TP + FN)
```

### Average Precision

AP is calculated from the precision-recall relationship across different confidence levels.

### mAP@50

The current implementation supports:

```text
IoU threshold = 0.50
```

and reports the mean AP across classes containing ground-truth objects.

### mAP@50:95

The implementation also evaluates:

```text
IoU = 0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
0.95
```

and averages the resulting mAP values.

---

## Current Evaluation Status

The evaluation implementation has been validated using controlled synthetic examples.

One synthetic evaluation produced:

```text
mAP@50      = 0.6667
mAP@50:95   = 0.4667
```

These numbers are **implementation-test results only**.

They should not be interpreted as the performance of CALD-Net.

No real-world detector benchmark is currently claimed.

---

## Synthetic Training

The project currently contains a synthetic detection dataset for testing the complete training pipeline.

The synthetic dataset is intended to verify:

- Forward propagation
- Target assignment
- Loss calculation
- Gradient propagation
- Optimizer updates
- DataLoader behavior
- Inference
- Evaluation

It is **not** intended to measure real detection performance.

---

## Hardware and Development Environment

Current development environment:

```text
OS:
Windows 11

Python:
3.14.7

PyTorch:
2.14.0+cu130

CUDA:
13.0

GPU:
NVIDIA GeForce RTX 3050 Ti Laptop GPU

GPU VRAM:
4 GB
```

The project is being developed with an emphasis on lightweight computation and compatibility with modest hardware.

---

## Research Direction

Potential research directions include:

### 1. Lightweight feature extraction

Investigate more efficient convolutional building blocks and feature extraction strategies.

### 2. Condition-aware feature modulation

Introduce a mechanism capable of adapting feature representations according to visual conditions such as:

- Darkness
- Low contrast
- Overexposure
- Noise
- Blur
- Backlighting

### 3. Lightweight global context

Investigate whether limited global-context information can improve detection in difficult scenes without introducing the computational cost of a large transformer-based backbone.

### 4. Efficient multi-scale fusion

Investigate improved feature fusion for objects with different scales while keeping the neck computationally efficient.

### 5. Knowledge distillation

Investigate whether a larger teacher detector can transfer useful information to the lightweight CALD-Net student.

### 6. Condition-specific evaluation

Instead of reporting only overall detection performance, evaluate the detector separately under different visual conditions.

### 7. Dynamic computation

Investigate whether computation can be selectively increased for difficult samples while keeping easy samples inexpensive.

These are research hypotheses, not established improvements. Each component will be validated experimentally.

---

## Planned Research Workflow

```text
Baseline
   │
   ▼
Real Dataset
   │
   ▼
Baseline Training
   │
   ▼
Baseline Benchmark
   │
   ├───────────────┐
   ▼               ▼
Architecture    Efficiency
Experiment      Measurement
   │               │
   └───────┬───────┘
           ▼
       Ablation Study
           │
           ▼
 Condition-Specific Evaluation
           │
           ▼
      Final CALD-Net
```

---

## Benchmarking Plan

Each major experiment should report both detection quality and computational cost.

### Detection metrics

```text
Precision
Recall
mAP@50
mAP@50:95
AP per class
```

### Efficiency metrics

```text
Parameter count
FLOPs
Inference latency
Throughput
GPU memory usage
Model size
```

### Robustness metrics

Where the dataset supports it:

```text
Normal conditions
Low light
Extreme darkness
Overexposure
Backlighting
Low contrast
Noise
Motion blur
Defocus blur
```

The purpose is to determine whether improvements in robustness justify their computational cost.

---

## Experiment Principles

The project follows these principles:

1. Establish a reproducible baseline before modifying the architecture.
2. Introduce major architectural changes independently where practical.
3. Measure computational cost alongside accuracy.
4. Keep training and evaluation procedures consistent between experiments.
5. Evaluate difficult visual conditions separately.
6. Use ablation studies to determine which components actually contribute to improvements.
7. Do not claim an improvement without controlled experimental evidence.
8. Preserve experiment history through Git.

---

## Git Development Strategy

The repository uses `main` as the stable development branch.

Meaningful development stages can use feature branches such as:

```text
feature/dataset-pipeline
feature/baseline-training
feature/condition-module
feature/lightweight-backbone
feature/knowledge-distillation
```

Research experiments can use branches such as:

```text
experiment/baseline
experiment/condition-module
experiment/feature-fusion
experiment/distillation
```

The goal is to keep architectural changes traceable and reproducible.

---

## Roadmap

### Phase 1 — Baseline

- [x] Project structure
- [x] Lightweight CNN backbone
- [x] FPN
- [x] Detection head
- [x] Target assignment
- [x] Detection loss
- [x] Synthetic dataset
- [x] DataLoader
- [x] Training pipeline
- [x] Bounding-box decoding
- [x] Confidence filtering
- [x] NMS
- [x] Inference pipeline
- [x] Precision / Recall evaluation
- [x] AP
- [x] mAP@50
- [x] mAP@50:95

### Phase 2 — Real Dataset

- [ ] Select appropriate detection dataset
- [ ] Implement dataset loader
- [ ] Implement annotation parser
- [ ] Establish train/validation/test splits
- [ ] Add dataset sanity checks
- [ ] Train the untouched baseline
- [ ] Record baseline metrics

### Phase 3 — Benchmarking

- [ ] Parameter count
- [ ] FLOPs
- [ ] Model size
- [ ] Inference latency
- [ ] GPU memory usage
- [ ] Baseline benchmark report

### Phase 4 — CALD-Net Architecture

- [ ] Investigate condition representation
- [ ] Implement condition-adaptive feature modulation
- [ ] Investigate lightweight global context
- [ ] Investigate efficient feature fusion
- [ ] Compare architectural variants
- [ ] Perform ablation studies

### Phase 5 — Robustness

- [ ] Low-light evaluation
- [ ] Extreme-darkness evaluation
- [ ] Overexposure evaluation
- [ ] Backlighting evaluation
- [ ] Low-contrast evaluation
- [ ] Noise evaluation
- [ ] Motion-blur evaluation
- [ ] Defocus-blur evaluation

### Phase 6 — Final Model

- [ ] Select validated components
- [ ] Finalize architecture
- [ ] Train final model
- [ ] Run complete benchmark
- [ ] Compare against baseline
- [ ] Perform final ablation study
- [ ] Document limitations
- [ ] Prepare research report / paper

---

## Research Notes

The `notes/` directory contains the project's ongoing research record.

Important design decisions, experiment results, observations, and architectural hypotheses should be recorded there as the project develops.

The goal is to maintain enough information to reproduce why a particular architectural decision was made.

---

## Disclaimer on Current Results

CALD-Net is an active research project.

The current baseline and synthetic experiments are intended to validate the implementation and development pipeline.

No claim of state-of-the-art performance is currently made.

Final conclusions will be based on experiments conducted on real datasets with controlled comparisons.

---

## License

License to be determined as the project progresses.
