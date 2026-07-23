# 🌾 Rice Guard

## AI-Powered Rice Disease Detection and Explainability Platform

Rice Guard is an end-to-end computer vision application designed to classify rice plant health conditions from uploaded images.

The platform uses a pretrained **MobileNetV3-Small** model to predict one of ten rice-health classes. It also provides prediction confidence, top-three predictions, uncertainty warnings, disease information, and Grad-CAM visual explanations.

The project combines image classification, transfer learning, model evaluation, explainable AI, and Streamlit deployment in one practical agricultural AI system.

---

## 📌 Project Overview

Rice diseases can significantly affect crop health, productivity, and food security. Early recognition of visible symptoms can help farmers and agricultural specialists investigate potential problems more quickly.

Rice Guard analyzes an uploaded rice image and provides:

- The most likely rice-health condition
- Prediction confidence
- Top-three model predictions
- An uncertainty warning for ambiguous results
- General information about the predicted condition
- Grad-CAM visual explanations
- Model-performance statistics

> Rice Guard is an educational AI project. Its predictions should not replace professional agricultural inspection or laboratory diagnosis.

---

## ✨ Main Features

### 🌿 Rice Disease Classification

Users can upload a rice image in one of the following formats:

- JPG
- JPEG
- PNG
- WebP

The trained model predicts one of ten rice-health classes.

### 📊 Prediction Confidence

The application displays:

- Most likely prediction
- Confidence percentage
- Top-three predictions
- Confidence gap between the first and second predictions

### ⚠️ Uncertainty Detection

The application displays an uncertainty warning when:

- Prediction confidence is below 60%, or
- The confidence difference between the top two predictions is below 10%

This prevents uncertain results from being presented as highly reliable diagnoses.

### 🔍 Grad-CAM Explainability

Grad-CAM is used to highlight image regions that influenced the model's prediction.

The application displays:

1. Original image
2. Attention heatmap
3. Grad-CAM overlay

Grad-CAM represents model attention and does not provide a biologically verified disease boundary.

### 📖 Condition Information

The application provides:

- A short description of the predicted condition
- A responsible suggested next step
- A reminder to seek professional agricultural guidance

### 📈 Model Performance Dashboard

The Streamlit application displays the final model's:

- Test accuracy
- Macro F1-score
- Weighted F1-score
- Test loss

---

## 🏷️ Rice-Health Classes

The model classifies the following ten conditions:

1. Bacterial Leaf Blight
2. Bacterial Leaf Streak
3. Bacterial Panicle Blight
4. Blast
5. Brown Spot
6. Dead Heart
7. Downy Mildew
8. Hispa
9. Normal
10. Tungro

---

## 📂 Dataset

The project uses the **Paddy Disease Classification** image dataset.

### Dataset summary

- Total labeled images: **10,407**
- Number of classes: **10**
- Image type: Rice plant and rice-leaf images

Because the original competition test labels were unavailable, the labeled dataset was divided using stratified sampling:

```text
Training set:   70%
Validation set: 15%
Test set:       15%
```

Stratified splitting helped preserve class proportions across all three subsets.

---

## 🔄 Project Workflow

```text
Data Audit
    ↓
Image Preprocessing
    ↓
Train/Validation/Test Split
    ↓
Baseline CNN Training
    ↓
Transfer Learning
    ↓
Model Comparison
    ↓
Detailed Evaluation
    ↓
Confidence and Error Analysis
    ↓
Grad-CAM Explainability
    ↓
Streamlit Application
```

---

## 🧠 Models Developed

Three image-classification models were trained and evaluated.

### 1. Baseline CNN

A custom convolutional neural network was trained from scratch to establish an initial performance benchmark.

### 2. MobileNetV3-Small

A pretrained MobileNetV3-Small model was adapted to the ten rice-health classes.

The convolutional feature extractor was initially frozen while the classification head was trained.

### 3. EfficientNet-B0

A pretrained EfficientNet-B0 model was also evaluated using transfer learning.

---

## 📊 Model Comparison

| Model | Best Epoch | Validation Accuracy | Test Loss | Test Accuracy | Macro F1 | Weighted F1 | Training Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline CNN | 10 | 57.27% | 1.2682 | 58.13% | 55.33% | 56.96% | 74.30 min |
| MobileNetV3-Small | 10 | 77.83% | 0.5967 | **78.17%** | **77.45%** | **78.47%** | 14.27 min |
| EfficientNet-B0 | 9 | 62.72% | 1.0603 | 62.93% | 61.00% | 62.75% | 15.22 min |

---

## 🏆 Final Model

**MobileNetV3-Small** was selected as the final model.

### Final performance

```text
Test Accuracy:      78.17%
Macro Precision:    76.84%
Macro Recall:       79.68%
Macro F1-Score:     77.45%
Weighted F1-Score:  78.47%
Test Loss:          0.5967
```

MobileNetV3-Small achieved:

- The highest test accuracy
- The highest macro F1-score
- The highest weighted F1-score
- The lowest test loss
- Significantly shorter training time than the baseline CNN

It provided the best balance between predictive performance and computational efficiency.

---

## 🔎 Model Evaluation

The final MobileNetV3-Small model was evaluated using:

- Overall test accuracy
- Class-level precision
- Class-level recall
- Class-level F1-score
- Confusion matrix
- Confidence analysis
- High-confidence error analysis
- Low-confidence correct-prediction analysis
- Most-confused class pairs
- Grad-CAM explanations

### Confidence Analysis

Correct and incorrect predictions were analyzed separately.

The application also compares the top two probabilities to identify uncertain cases.

### Error Analysis

Important errors were reviewed, including:

- High-confidence incorrect predictions
- Low-confidence correct predictions
- Frequently confused class pairs
- Visually similar disease symptoms

This analysis helps reveal where the model may require additional training data or improvement.

---

## 🔥 Explainable AI with Grad-CAM

Grad-CAM uses feature activations and prediction gradients from the final convolutional feature layer.

The generated heatmap highlights image regions that had greater influence on the selected prediction.

The Streamlit dashboard displays:

```text
Original Image
      ↓
Grad-CAM Heatmap
      ↓
Grad-CAM Overlay
```

### Important limitation

Grad-CAM does not prove that a highlighted region contains a disease symptom.

It only indicates which regions influenced the model's prediction.

---

## 🖥️ Streamlit Application

The application includes:

- Responsive wide-page layout
- Custom agricultural theme
- Image uploader
- WebP support
- Prediction result cards
- Confidence warning system
- Top-three prediction table
- Condition-information section
- Grad-CAM visualization
- Model-performance metrics
- Responsible AI disclaimer

---

## 📁 Project Structure

```text
16_Rice_Guard/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── train_images/
│   ├── train.csv
│   ├── validation_split.csv
│   ├── test_split.csv
│   ├── label_mapping.json
│   └── class_weights.csv
│
├── models/
│   ├── baseline_cnn_best.pth
│   ├── mobilenet_v3_small_stage1_best.pth
│   └── efficientnet_b0_best.pth
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_image_preprocessing.ipynb
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_transfer_learning.ipynb
│   └── 05_evaluation_explainability.ipynb
│
└── outputs/
    ├── baseline_metrics.csv
    ├── baseline_classification_report.csv
    ├── baseline_test_predictions.csv
    │
    ├── mobilenet_metrics.csv
    ├── mobilenet_classification_report.csv
    ├── mobilenet_test_predictions.csv
    │
    ├── efficientnet_metrics.csv
    ├── efficientnet_classification_report.csv
    ├── efficientnet_test_predictions.csv
    │
    ├── model_comparison.csv
    │
    └── evaluation_explainability/
        ├── final_model_predictions.csv
        ├── final_classification_report.csv
        ├── confidence_summary.csv
        ├── class_confidence_analysis.csv
        ├── high_confidence_errors.csv
        ├── low_confidence_correct_predictions.csv
        ├── most_confused_class_pairs.csv
        ├── gradcam_examples.png
        ├── gradcam_samples.csv
        └── final_evaluation_summary.csv
```

Some filenames may vary slightly depending on the saved local project structure.

---

## 🛠️ Technologies Used

### Programming and Data Processing

- Python
- NumPy
- Pandas

### Deep Learning

- PyTorch
- Torchvision
- Convolutional Neural Networks
- Transfer Learning
- MobileNetV3-Small
- EfficientNet-B0

### Model Evaluation

- Scikit-learn
- Classification Report
- Confusion Matrix
- Precision, Recall and F1-Score

### Explainability

- Grad-CAM
- PyTorch Forward Hooks
- Gradient-Based Feature Visualization

### Visualization and Application

- Matplotlib
- Pillow
- Streamlit

### Development Tools

- Visual Studio Code
- Jupyter Notebook
- Google Colab
- Git
- GitHub

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
```

Move into the project directory:

```bash
cd YOUR_REPOSITORY_NAME
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📦 Requirements

The application requires:

```text
streamlit
torch
torchvision
numpy
pandas
Pillow
matplotlib
```

---

## ▶️ Run the Application

From the project root directory, run:

```bash
streamlit run app.py
```

Streamlit will open the application in your browser.

The local address is usually:

```text
http://localhost:8501
```

---

## 🧪 How to Use Rice Guard

1. Open the Streamlit application.
2. Upload a rice image.
3. Wait for the model to process the image.
4. Review the most likely prediction.
5. Check the confidence score.
6. Compare the top-three predictions.
7. Read the uncertainty warning when displayed.
8. Review the condition information.
9. Examine the Grad-CAM heatmap and overlay.
10. Seek professional agricultural confirmation when necessary.

---

## 📷 Application Preview

Add your final application screenshot here:

```markdown
![Rice Guard Application](assets/rice_guard_dashboard.png)
```

Recommended screenshot content:

- Rice Guard title
- Uploaded image
- Prediction result
- Confidence score
- Top predictions
- Grad-CAM explanation
- Model-performance cards

---

## ⚠️ Limitations

Rice Guard has several important limitations:

- The model was trained using a limited labeled dataset.
- Performance may decrease for images captured under different conditions.
- Background objects may influence predictions.
- Similar-looking diseases may be confused.
- Confidence does not guarantee correctness.
- The model recognizes only the ten trained classes.
- A normal prediction does not guarantee that a plant is disease-free.
- Grad-CAM does not provide verified symptom segmentation.
- The system has not been validated for professional agricultural diagnosis.

---

## 🚀 Future Improvements

Potential improvements include:

- Collecting more images from different farms and environments
- Improving class balance
- Fine-tuning additional MobileNet layers
- Testing stronger image augmentation
- Adding confidence calibration
- Adding out-of-distribution image detection
- Improving background removal
- Adding multilingual support
- Adding mobile-camera support
- Exporting the model to ONNX or TensorFlow Lite
- Validating predictions with agricultural specialists
- Adding location and weather context in a future version

---

## 🌱 Responsible AI Statement

Rice Guard is designed as an educational decision-support prototype.

The application should not be used as the only source for disease diagnosis, pesticide selection, treatment decisions, or crop-management actions.

Predictions should be confirmed through:

- Field inspection
- Local agricultural experts
- Plant pathology specialists
- Laboratory testing when appropriate

---

## 👩‍💻 Author

**Shwe Yamin**

Data Science Graduate  
Data Analysis · Machine Learning · Computer Vision
---

## 📄 License

This project is intended for educational and portfolio purposes.

Dataset ownership and usage rights remain with the original dataset provider.

---

## ⭐ Support

If you find this project useful, consider giving the repository a star.

```text
Rice Guard
AI-Powered Rice Disease Detection and Explainability Platform
```