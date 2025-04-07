# TGAPneumoSeg
TGAPneumoSeg: A Text-Guided Attention Network for Pneumothorax Segmentation in Chest X-rayss


This repository contains code and documentation for **TGANet-based Pneumothorax Image Segmentation**. The core model implementation resides in the `main.ipynb` notebook.



## 📌 Objective

To design and implement a deep learning model based on **TGANet architecture** for the segmentation of **pneumothorax** in chest X-ray images, targeting high segmentation accuracy on the **Hyper-Kvasir Segmented Dataset** and achieving state-of-the-art performance.

## 🧠 Model Architecture

The TGA (Temporal-Gated Attention) network integrates:
- **Temporal attention modules**
- **Spatial feature encoders**
- **Gating mechanisms**

This combination allows the model to extract rich spatio-temporal features, which are particularly useful for accurately detecting and segmenting pneumothorax regions.

## 🧪 Dataset

- **Dataset**: Hyper-Kvasir Segmented Dataset
- **Data Size**: ~12,000 annotated X-ray images
- **Format**: Image and corresponding binary mask

## 🧾 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt

```
Updates are underway...
