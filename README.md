
# Predicting the shifts of spinal cord in Head and Neck cancer patients receiving radiotherapy


This repository contains the official code and resources for the study on deep learning–based spinal cord segmentation and positional shift prediction from CT scans, designed to enhance adaptive radiotherapy workflows for head and neck cancer patients. The project investigates the integration of U-Net–based segmentation with a CNN–LSTM prediction model to automate and refine spinal cord tracking during treatment.

---

## Abstract  
Accurate spinal cord segmentation and monitoring are critical for safe and effective adaptive radiotherapy. Manual contouring is time-consuming and subject to inter-observer variability, which can compromise treatment precision. This project presents an automated pipeline:  
- A U-Net architecture performs pixel-level segmentation of the spinal cord from CBCT scans.  
- The segmented spinal cord regions are analyzed by a CNN–LSTM prediction model to estimate positional shifts over the treatment course.  

The approach achieved Dice scores exceeding 0.85 for segmentation and mean absolute errors of 1–2 mm for shift prediction, demonstrating strong potential for clinical integration.  

---

## Model Architecture  

### Segmentation Model (U-Net)  
- **Encoder**: Multi-level convolutional layers with ReLU activations and max pooling to capture contextual features.  
- **Decoder**: Transposed convolutions with skip connections to restore spatial resolution and preserve fine-grained details.  
- **Output**: Sigmoid-activated pixel-wise classification mask for spinal cord delineation.  
- **Loss**: Dice loss combined with binary cross-entropy for robust optimization under class imbalance.  

### Prediction Model (CNN–LSTM)  
- **CNN Feature Extractor**: Convolutional blocks (Conv2D, ReLU, MaxPool) encode spatial features of segmented spinal cord slices.  
- **LSTM Layers**: Capture temporal dependencies across sequential CBCT scans to track positional shifts.  
- **Fully Connected Layer**: Predicts 2D displacement vectors (x, y) for spinal cord movement.  
- **Calibration**: Outputs validated against expert landmarks to ensure clinically relevant measurements.  

---

## Repository Structure  

SpinalCord-Shift-Prediction/

├── Model.ipynb               # Training pipeline for U-Net and CNN–LSTM

├── metrics/                  # Loss, accuracy, Dice/IoU curves (PNG)
│   ├── all\_training\_metrics.png
│   ├── loss\_accuracy\_curve.png
│   └── loss\_curve.png

├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md



---

## Setup and Usage  

### 1. Prerequisites  
- Python 3.8+  
- PyTorch (with CUDA for GPU acceleration)  
- NumPy, OpenCV, Matplotlib, and related libraries  

### 2. Installation  
Clone the repository:  
```bash
git clone https://github.com/RedEgnival/spinal-cord-shift.git
cd SpinalCord-Shift-Prediction
````

Create and activate a virtual environment:

```bash
python -m venv venv  
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Data Structure

Due to privacy, patient CT scans are not included. Organize your dataset as follows:

```
data/
├── patient_01/
│   ├── CBCT001.dcm
│   ├── ...
│   └── RS.dcm        # Ground truth contours
├── patient_02/
│   └── ...
└── patient_N/
```

### 4. Running the Pipeline


1. **Model.ipynb** – Preprocess Dicom files then train the U-Net segmentation model and CNN–LSTM shift predictor.


---

## Benchmarks

* **Segmentation**: Dice coefficient > 0.85 on validation, stable IoU trends across epochs.
* **Prediction**: Mean absolute error \~1–2 mm against expert measurements.
* **Cross-validation**: Confirmed robustness across multiple patients.
* **Training Curves**: Included in `metrics/` for reproducibility.

---

## Citation

If you use this repository or its methods, please cite the following works:

(A full citation will be provided once the related manuscript is published.)

---

## Acknowledgments

This project is a collaboration between SVKM’s NMIMS University, Mukesh Patel School of Technology, Management & Engineering, Mumbai, and Nanavati Max Super Speciality Hospital, Mumbai. Special thanks to the clinical experts for providing guidance and validation for model evaluation.

```
```
