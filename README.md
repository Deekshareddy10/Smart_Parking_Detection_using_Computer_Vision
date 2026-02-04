# 🚗 Smart Parking Detection Using Computer Vision

**Author:** Deeksha Reddy Patlolla  
**Institution:** University of Colorado Denver - MS Computer Science  
**Advisor:** Assistant Professor Mazen Al Borno

## Overview

This application provides real-time parking occupancy detection using computer vision. It combines **YOLOv8** for vehicle detection with **ROI-based slot classification** for accurate parking space analysis.

## Features

- **Automatic Dataset Detection**: Recognizes PKLot, CNRPark-EXT, or unknown image sources
- **Dual Processing Modes**:
  - **PKLot Mode**: ROI-based slot occupancy classification with polygon annotations
  - **Detection Mode**: Vehicle counting and localization for CNRPark-EXT or unknown datasets
- **Real-time Processing**: Fast inference using YOLOv8 nano model
- **Interactive UI**: Built with Streamlit for easy deployment on Hugging Face Spaces
- **Configurable Parameters**: Adjustable confidence thresholds and IoU settings

## Supported Datasets

### PKLot Dataset
- Over 12,000 labeled images from three camera viewpoints (UFPR04, UFPR05, PUCPR)
- Includes XML annotations with parking slot polygons
- Various weather conditions: sunny, cloudy, rainy

### CNRPark-EXT Dataset
- Over 150,000 images with challenging conditions
- Multiple camera angles
- Glare, shadows, fog, and nighttime scenarios

## Technical Approach

1. **Vehicle Detection**: YOLOv8 identifies vehicles (cars, trucks, buses, motorcycles)
2. **ROI Scaling**: PKLot polygon annotations are scaled to match input image resolution
3. **Occupancy Classification**: Center-point and IoU-based heuristics determine slot status
4. **Visualization**: Color-coded overlays show occupied (red) and empty (green) slots

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Local Development
```bash
streamlit run app.py
```

### Hugging Face Spaces
Upload all files to a new Space with the Streamlit SDK.

## File Structure

```
smart_parking/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── sample_rois/       # Sample PKLot XML files (optional)
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| Detection Confidence | Minimum confidence for vehicle detection | 0.25 |
| IoU Threshold | Overlap threshold for occupancy classification | 0.15 |
| Show Debug Info | Display detailed detection information | False |

## References

1. Almeida et al., "PKLot: A Robust Dataset for Parking Lot Classification" (2015)
2. Amato et al., "Deep Learning for Decentralized Parking Lot Occupancy Detection" (2017)
3. Ultralytics, "YOLOv8: Real-Time Object Detection and Segmentation" (2023)

# Deployment
https://huggingface.co/spaces/Deekshareddy10/XML_blend
https://huggingface.co/spaces/Deekshareddy10/smart-parking-detection


## License

Copyright © 2025 Deeksha Reddy Patlolla. All Rights Reserved.

---

