# TGAPneumoSeg

**TGAPneumoSeg: A Text-Guided Attention Network for Pneumothorax Segmentation in Chest X-rays**

This repository provides the implementation of **TGAPneumoSeg**, a novel deep learning architecture designed to accurately segment pneumothorax regions in chest radiographs using a **text-guided attention mechanism**. The project leverages multi-scale attention, text embeddings, and progressive denoising techniques to achieve state-of-the-art performance on the **SIIM-ACR Pneumothorax Dataset**.

## 📌 Objective

To design and develop a **text-guided deep learning model** capable of robust and accurate segmentation of pneumothorax lesions, enabling efficient and clinically relevant diagnosis using chest X-rays.

## 🧠 Model Architecture: TGAPneumoSeg

TGAPneumoSeg is a modification of TGANet designed for medical image segmentation. It integrates:

* **SE-ResNeXt50-based Encoder** for hierarchical spatial feature extraction
* **Feature Enhancement Module (FEM)** with dilated convolutions and dual attention
* **Text-guided Embedding Fusion** that contextualizes visual features with lesion attributes
* **Progressive Denoising Attention (PDA)** stages to refine noisy medical image features
* **Multi-scale Decoder with Label Attention Mechanism** for segmentation output

### 🔁 Model Workflow

```
+-------------------+        +---------------------+        +--------------------+
| Chest X-ray Image | -----> | SE-ResNeXt Encoder  | -----> | FEM (Multi-scale)  |
+-------------------+        +---------------------+        +--------------------+
                                                             |
                                                             v
                                        +--------------------------------+
                                        | Text-guided Embedding Fusion   |
                                        +--------------------------------+
                                                             |
                                                             v
                                     +------------------------------------+
                                     | PDA-1 + PDA-2 (Denoising Layers)   |
                                     +------------------------------------+
                                                             |
                                                             v
                                     +----------------------------------+
                                     | Decoder + Label Attention Module |
                                     +----------------------------------+
                                                             |
                                                             v
                                               +---------------------+
                                               | Segmentation Output |
                                               +---------------------+
```

## 🗃 Dataset: SIIM-ACR Pneumothorax Segmentation

* **Total Images**: 12,047 chest X-ray images
* **With Masks**: 2,669 images
* **Format**: DICOM (converted to PNG)
* **Split**:

  * Training Set: 9,637 (2,153 with mask)
  * Validation Set: 2,410 (516 with mask)

## 🔧 Preprocessing Pipeline

* Convert DICOM → PNG
* Resize to **256x256**
* Apply **Albumentations** for augmentation:

  * Rotation (±15°), Horizontal Flip
  * Brightness/Contrast Adjustment
  * CLAHE (Contrast Limited Adaptive Histogram Equalization)
* Generate **textual labels**: lesion count and size (zero, one, multiple; small, medium, large)
* Extract bounding boxes for text embedding guidance

## 🧬 Textual Embedding

* Generated using `Text2Embed` module with **BPEmb**
* Embeddings represent lesion **count** and **size**
* Embedded features fused with visual features to enhance segmentation accuracy

## 🔍 Components Breakdown

### 🧩 Encoder (SE-ResNeXt50\_32x4d)

* 5 stages with increasing depth
* Global average pooling + squeeze-and-excitation (SE) blocks
* Produces multiscale feature representations E = {e1, ..., e5}

### 🎯 Feature Enhancement Module (FEM)

* Uses dilated convolutions (rates: 1, 6, 12, 18)
* Channel and Spatial Attention modules

### 🧠 Text-Guided Attention & Embedding Fusion

* Predicts lesion size & count → embeds them → fuses with high-level visual features

### 🌀 Two-Stage Progressive Denoising Attention (PDA)

1. **PDA-1**: Residual conv layers for feature denoising
2. **PDA-2**: U-Net-based refinement block

### 🧩 Decoder

* Upsamples fused features with **skip connections**
* Uses **Label Attention** to weight features based on textual predictions
* Final segmentation via sigmoid-activated 1x1 conv layer

## ⚙️ Training Details

| Parameter            | Value                               |
| -------------------- | ----------------------------------- |
| Input Size           | 256 x 256                           |
| Batch Size           | 8                                   |
| Epochs               | 50                                  |
| Optimizer            | AdamW                               |
| Learning Rate        | 0.0001                              |
| Loss Function        | Dice + BCE + Focal Loss             |
| Attention Mechanisms | Channel + Spatial + Label Attention |
| Encoder Backbone     | SE-ResNeXt50                        |
| Mixed Precision      | Enabled (AMP)                       |

## 📈 Results

| Model                    | Dice Score | Jaccard Index |
| ------------------------ | ---------- | ------------- |
| U-Net                    | 0.6934     | 0.6758        |
| ResNet34 U-Net           | 0.7953     | 0.7725        |
| Attention U-Net          | 0.8261     | 0.8116        |
| Attention Residual U-Net | 0.8314     | 0.8197        |
| PTXSeg-Net               | 0.8346     | 0.8194        |
| **TGAPneumoSeg (Ours)**  | **0.8394** | **0.8198**    |

### 📉 Loss and Metric Curves

* Training/Validation Dice Score
* Training/Validation Jaccard Index
* Training/Validation Loss

![Training Curves](./assets/training_curves.png)

### 🧪 Sample Results

| Input X-ray             | Ground Truth         | TGAPneumoSeg Prediction      |
| ----------------------- | -------------------- | ---------------------------- |
| ![](./assets/input.png) | ![](./assets/gt.png) | ![](./assets/prediction.png) |

## 📌 Conclusion

TGAPneumoSeg significantly improves pneumothorax segmentation by combining spatial features with clinically interpretable text-guided cues. The dual denoising strategy and advanced attention modules contribute to superior performance.

## 🚀 Future Scope

* Expand to multimodal segmentation (e.g., CT, MRI)
* Apply to other medical tasks: brain tumors, lung nodules
* Real-world clinical validation in hospitals
* Optimization for edge deployment (mobile radiology units)

## 📚 References

citation to be uploaded..
