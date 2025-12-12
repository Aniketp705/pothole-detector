# Pothole Detector 🕳️🚗

A deep learning project designed to detect potholes in road images using Transfer Learning with **MobileNetV2**. This model helps in identifying road damages efficiently, which can be useful for road maintenance and safety systems.

## 📌 Features

- **Automated Data Splitting:** Automatically splits the dataset into Training (80%) and Testing (20%) sets.
- **Data Augmentation:** Uses `ImageDataGenerator` for rotation and flipping to improve model generalization.
- **Transfer Learning:** Utilizes the pre-trained **MobileNetV2** model (weights from ImageNet) for feature extraction.
- **High Accuracy:** Achieves ~93% accuracy on the test set.
- **Visualization:** Generates a Confusion Matrix and Classification Report for performance evaluation.
- **Inference Script:** Allows users to upload and test new images for pothole detection.

## 🛠️ Tech Stack

- **Language:** Python 3
- **Deep Learning Framework:** TensorFlow / Keras
- **Computer Vision:** OpenCV (for image processing if needed)
- **Data Manipulation:** NumPy, Shutil, OS
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning Metrics:** Scikit-learn

## 📂 Dataset Structure

The project expects the source dataset to be organized as follows:

```
/content/dataset/
    ├── plain/       # Images of normal roads
    └── potholes/    # Images of roads with potholes
```

The script automatically processes this into:

```
/content/processed_data/
    ├── train/
    │   ├── plain/
    │   └── potholes/
    └── test/
        ├── plain/
        └── potholes/
```

## 🚀 Installation & Usage

### 1. Prerequisites
Ensure you have the required libraries installed. You can install them via pip:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### 2. Running the Notebook
1.  Clone this repository or download the `Potholes.ipynb` file.
2.  Open the notebook in **Google Colab** or **Jupyter Notebook**.
3.  Ensure your dataset is uploaded and the `SOURCE_PATH` variable in the notebook matches your dataset location.
4.  Run all cells to:
    - Split the data.
    - Train the MobileNetV2 model.
    - Evaluate the model.
    - Save the trained model as `pothole_detector_final.h5`.

### 3. Testing with New Images
The notebook includes a section to upload and test custom images. It will classify the image as either:
- **POTHOLE DETECTED 🚨** (Red Label)
- **Road is Safe ✅** (Green Label)

## 📊 Model Performance

- **Base Model:** MobileNetV2 (frozen backbone)
- **Optimizer:** Adam (learning rate = 0.0005)
- **Loss Function:** Binary Crossentropy
- **Test Accuracy:** ~93.71%

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## 📜 License

This project is open-source and available for educational purposes.
