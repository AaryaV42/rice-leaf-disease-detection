

---

# **Rice Leaf Disease Detection**

This project builds a complete system for detecting diseases in rice leaves using machine learning and image processing. It covers data preparation, feature engineering, model training, evaluation, and saving the final model. The main implementation is inside the Jupyter Notebook `rice leaf disease detection.ipynb`.

## **Project Overview**

Rice crops often suffer from bacterial and fungal diseases. Early and accurate detection helps prevent large-scale damage. This project automates disease classification by analyzing leaf images.

The system performs:

* Loading and preprocessing images
* Normalization and augmentation
* Feature extraction using CNN or traditional methods
* Training models such as Random Forest, CNN, SVM, or a stacking ensemble
* Evaluating performance
* Saving the best model for deployment

## **Project Structure**

```
rice-leaf-disease-detection
 ├── rice leaf disease detection.ipynb
 ├── README.md
 ├── /dataset
 │    ├── Healthy/
 │    ├── Diseased/
 │    └── ...
 ├── /models
 │    └── final_model.pkl or .onnx
 ├── /results
 │    ├── accuracy.png
 │    ├── confusion_matrix.png
 │    └── sample_predictions/
 └── requirements.txt
```

## **Features Implemented**

### **1. Image Preprocessing**

* Resizing
* Normalization
* Converting to numeric arrays
* Train-test split

### **2. Data Augmentation (optional)**

* Rotations
* Zoom
* Horizontal and vertical flips

### **3. Model Development**

Models used may include:

* CNN (TensorFlow or PyTorch)
* Random Forest
* SVM
* Stacking ensemble
* Export to ONNX for deployment

### **4. Model Evaluation**

* Accuracy, precision, recall, F1-score
* Confusion matrix
* Training vs validation curves

### **5. Model Saving**

* Using pickle, joblib, or ONNX

## **Getting Started**

### **Prerequisites**

Install all required dependencies:

```
pip install -r requirements.txt
```

Common libraries used:

* numpy
* pandas
* matplotlib
* scikit-learn
* tensorflow or torch
* opencv-python
* onnx / onnxruntime

## **How to Run the Project**

1. Open the notebook:

```
jupyter notebook "rice leaf disease detection.ipynb"
```

2. Run each cell in order.
3. The trained model will be saved inside the `/models` folder.
4. Update dataset paths in the notebook if needed.

## **Results**

The notebook generates:

* Accuracy score
* Confusion matrix
* Sample predictions
* Plot of training vs validation performance

You can find these inside the **results** folder.

## **Future Improvements**

* Deploy using Flask, FastAPI, or Streamlit
* Add mobile inference support using ONNX
* Add more disease categories
* Try Vision Transformers (ViT) for better accuracy

## **License**

This project is available for academic and research use.

## **Acknowledgements**

* Dataset providers (Kaggle or other sources)
* OpenCV, TensorFlow, scikit-learn communities

---


