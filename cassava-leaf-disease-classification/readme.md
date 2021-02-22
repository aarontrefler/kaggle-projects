# Cassava Leaf Disease Classification

[Kaggle Competition Link](https://www.kaggle.com/c/cassava-leaf-disease-classification)  
* Submission Date: 02-18-2021  
* Score: 2708th / 3900 (Top 70%)

## Overview
* **Goal**: Classify images of Cassava leaves into four disease categories or a healthy category
* **Preprocessing**: image processing was performed using fast.ai Data Loaders at the image (random cropping) and batch level (lip, rotate, zoom, warp, lighting transforms)
* **Modeling**: Transfer learning was applied to a EfficientNet-B3 convolutional neural network model using the fast.ai.

## Project
---
```
├── working      <- Jupyter submission notebook
├── readme.md
```