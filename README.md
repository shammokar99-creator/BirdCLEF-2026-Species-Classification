# BirdCLEF 2026: Acoustic Species Identification

This repository contains my solution for the BirdCLEF 2026 Kaggle competition. The goal is to identify bird vocalizations in complex, polyphonic soundscapes from the Pantanal region using Deep Learning.

## Project Overview
Working with audio data presents unique challenges like overlapping bird calls and environmental noise. My approach treats audio as an image classification problem by converting recordings into Mel-spectrograms.

## Tech Stack & Methodology
- **Model:** EfficientNet-B0 (via `timm` library).
- **Data Processing:** - Audio segmented into 5-second chunks.
  - Conversion to Mel-spectrograms ($n\_fft=2048$, $hop\_length=512$).
- **Training:** 3 Epochs on Kaggle P100 GPU.
- **Optimization:** Adam optimizer with Cosine Annealing learning rate scheduling.

## Resilience Features
- **Hidden Test Inference:** Implemented a robust try-except fallback loop to ensure the notebook runs smoothly during private re-runs on hidden test data.
- **Offline Compatibility:** Configured for no-internet submission environments.

## Repository Structure
- `BirdCLEF2026.ipynb`: Main notebook containing preprocessing, training, and inference.
- `LICENSE`: MIT License.
