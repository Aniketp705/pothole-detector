# Pothole Detector 🕳️🚗

A deep learning project designed to detect potholes in road images using Transfer Learning with **MobileNetV2**. This project includes both a model training notebook and a real-time **Streamlit Web Application** for easy inference.

## 📌 Features

- **Interactive Web App:** User-friendly interface built with Streamlit for real-time pothole detection.
- **Automated Data Splitting:** Automatically splits the dataset into Training (80%) and Testing (20%) sets.
- **Data Augmentation:** Uses `ImageDataGenerator` for rotation and flipping to improve model generalization.
- **Transfer Learning:** Utilizes the pre-trained **MobileNetV2** model (weights from ImageNet) for feature extraction.
- **High Accuracy:** Achieves ~93% accuracy on the test set.
- **Visualization:** Generates a Confusion Matrix and Classification Report for performance evaluation.

## 🛠️ Tech Stack

- **Language:** Python 3
- **Web Framework:** Streamlit
- **Deep Learning Framework:** TensorFlow / Keras
- **Computer Vision:** OpenCV, PIL
- **Data Manipulation:** NumPy, Shutil, OS
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning Metrics:** Scikit-learn

## 📂 Project Structure

```
/
├── app.py                      # Streamlit Web Application
├── Potholes.ipynb              # Model Training Notebook
├── pothole_detector_final.h5   # Trained Model File
├── requirements.txt            # Project Dependencies
├── README.md                   # Project Documentation
└── .gitignore
```

## 🚀 Installation & Usage

### 1. Prerequisites
Ensure you have Python installed. Install the required dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Running the Web App
To start the interactive pothole detection interface:

1.  Ensure you have the trained model `pothole_detector_final.h5` in the project directory. (If not, run the notebook first to generate it).
2.  Run the Streamlit app:

```bash
streamlit run app.py
```

3.  The app will open in your browser. Upload an image to check if a pothole is detected!

### 3. Training the Model (Optional)
If you want to retrain the model or understand the training process:
1.  Open `Potholes.ipynb` in **Google Colab** or **Jupyter Notebook**.
2.  Follow the instructions to upload your dataset and run the cells.
3.  The notebook will save the new model as `pothole_detector_final.h5`.

## 📊 Model Performance

- **Base Model:** MobileNetV2 (frozen backbone)
- **Optimizer:** Adam (learning rate = 0.0005)
- **Loss Function:** Binary Crossentropy
- **Test Accuracy:** ~93.71%

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## 📜 License

This project is open-source and available for educational purposes.
