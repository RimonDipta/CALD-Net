# CALD-Net Research Notes

## Project

**Working name:** CALD-Net\
**Full idea:** Condition-Adaptive Lightweight Detector

### Research goal

Build a lightweight object detector that: - uses fewer computational
resources than strong detector baselines, - maintains high detection
accuracy, - remains robust under difficult visual conditions, - adapts
its feature processing according to image conditions.

### Target conditions

-   Normal illumination
-   Low light
-   Extreme darkness
-   Overexposure / saturation
-   Backlighting
-   Low contrast
-   Noise
-   Motion blur
-   Defocus blur

## Reference Paper

**E-TomatoDet --- Sun et al., BMC Plant Biology (2025)**

The paper starts from YOLOv8n and adds: - CSWinTransformer (CSWinT)
backbone components - Local Feature Enhance Pyramid (LFEP) -
CSP-Comprehensive Multi-Kernel Module (CSP-CMKM)

Reported tomato-disease results: - mAP50: 97.2% - mAP50-95: 80.2%

The paper also reports 8.9 ms inference and identifies lightweight
deployment as future work.

## Our intended research direction

Instead of copying E-TomatoDet, investigate:

1.  Lightweight CNN backbone
2.  Limited global-context processing
3.  Condition encoder
4.  Condition-adaptive feature modulation
5.  Lightweight multi-scale feature fusion
6.  Knowledge distillation from a stronger teacher
7.  Robustness evaluation by image condition
8.  Optional dynamic computation for difficult images

## Core hypothesis

A small detector can retain strong accuracy if it: - learns useful local
and global representations efficiently, - receives explicit information
about image condition, - allocates feature-processing capacity according
to condition, - learns from a stronger teacher.

This is a hypothesis. It must be experimentally validated.

------------------------------------------------------------------------

# Learning Log

## Stage 0 --- Environment

**Status:** Not started

Need to learn: - Python virtual environments - PyTorch - tensors -
GPU/CUDA basics - OpenCV

## Stage 1 --- Computer Vision Fundamentals

Topics: - image tensors - RGB/BGR - grayscale - HSV - luminance -
brightness - contrast - saturation - histograms - gamma - noise - blur -
image normalization

## Stage 2 --- Deep Learning

Topics: - tensor - neural network - convolution - feature map -
activation - loss - optimizer - backpropagation - training vs validation
vs test

## Stage 3 --- Object Detection

Topics: - bounding boxes - IoU - precision - recall - AP - mAP50 -
mAP50-95 - NMS - detection head - backbone - neck

## Stage 4 --- Baseline

Build and evaluate a small YOLO detector first.

Purpose: **Never claim our architecture is better until we have a
reproducible baseline.**

## Stage 5 --- CALD-Net

Implement modules one at a time: 1. Lightweight backbone 2. Condition
encoder 3. Feature modulation 4. Multi-scale fusion 5. Distillation 6.
Optional dynamic computation

## Stage 6 --- Experiments

Every experiment should record: - model version - dataset version -
training configuration - parameters - FLOPs - latency - GPU/CPU -
mAP50 - mAP50-95 - precision - recall - condition-specific results

------------------------------------------------------------------------

# Notes: Concepts to Remember

### Object detection

Object detection answers: \> What objects are present, where are they,
and what class does each belong to?

### Classification

Classification answers: \> What is in this image?

### Detection

Detection answers: \> What is in this image AND where is it?

### Backbone

Extracts visual features from the image.

### Neck

Combines features from different scales.

### Head

Produces the final object predictions.

### IoU

Intersection over Union measures overlap between predicted and
ground-truth boxes.

### Precision

Of predicted objects, how many were correct?

### Recall

Of real objects, how many were detected?

### mAP

Mean Average Precision; a standard detection performance metric.

------------------------------------------------------------------------

# Experiment Rule

Never change multiple major components and then claim which component
caused an improvement.

Prefer:

Baseline → + Module A → + Module B → + Module C → Final model

This is called an **ablation study**.

------------------------------------------------------------------------

# Research Integrity

Always separate: - measured results, - assumptions, - hypotheses, -
literature claims.

Do not claim "higher accuracy" or "lower compute" until experiments
demonstrate it.

------------------------------------------------------------------------

# Experiment Log

  -----------------------------------------------------------------------------------------------
  ID        Model   Dataset   Condition     Params    FLOPs   Latency    mAP50   mAP50-95 Notes
  --------- ------- --------- ----------- -------- -------- --------- -------- ---------- -------
  EXP-001   ---     ---       ---              ---      ---       ---      ---        --- ---

  -----------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# Questions / Ideas

-   How small can the backbone become before accuracy collapses?
-   Which conditions cause the largest performance drop?
-   Does explicit condition information improve detection?
-   Is enhancement better than direct condition-aware detection?
-   Does knowledge distillation recover lost accuracy?
-   Can difficult images trigger additional computation without slowing
    normal images?
-   How does the model perform on unseen conditions?

------------------------------------------------------------------------

# Current Step

**STEP 0: Set up the ML environment and verify GPU acceleration.**

Do not start model architecture work until the environment is
reproducible.
