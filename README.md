# 🚢 Titanic Passenger Survival Prediction with 1 Million Dataset using Deep Learning (ANN)

A Deep Learning web application that predicts whether a passenger would survive the Titanic disaster based on passenger information.

The project demonstrates an end-to-end Machine Learning workflow including data preprocessing, feature engineering, model training using TensorFlow/Keras, model serialization, and deployment using Streamlit.


# 📌 Project Overview

The Titanic disaster is one of the most well-known historical datasets in Machine Learning. The objective of this project is to predict passenger survival using an Artificial Neural Network (ANN).

The application allows users to enter passenger details through an interactive Streamlit interface and instantly predicts the survival probability.

---

# Resource I used
> Dataset Name: Titanic Huge Dataset - 1M Passengers

> Dataset Link: https://www.kaggle.com/datasets/marcpaulo/titanic-huge-dataset-1m-passengers

---

# ✨ Features

- Interactive Streamlit Dashboard
- Deep Learning (Artificial Neural Network)
- Real-time Prediction
- Survival Probability
- Clean User Interface
- Data Preprocessing Pipeline
- Feature Scaling
- Label Encoding
- One-Hot Encoding
- Saved Model Loading (.h5)
- Production-ready Prediction Pipeline

---

# 📂 Project Structure

```
Titanic-Survival-Prediction/
│
├── app.py
├── model.h5
├── scaler.pkl
├── label_encoder.pkl
├── onehot_encoder.pkl
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── Titanic_ANN.ipynb
|   └── predict.ipynb
│
├── dataset/
│   └── Huge_1M_titanic.csv
│
└── images/
    └── dashboard.png
```

---

# 📊 Dataset

Dataset contains passenger information including:

- Passenger Class
- Gender
- Number of Siblings/Spouse
- Number of Parents/Children
- Fare
- Embarked Port

Target Variable

```
Survived
```

- 1 → Survived
- 0 → Did Not Survive

---

# 🧠 Model Architecture

Artificial Neural Network (ANN)

```
Input Layer

↓

Dense Layer (128)
ReLU

↓

Dense Layer (64)
ReLU

↓

Dense Layer (32)
ReLU

↓

Output Layer (1)
Sigmoid
```

Loss Function

```
Binary Cross Entropy
```

Optimizer

```
Adam
```

Evaluation Metric

```
Accuracy
```

---

# ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Label Encoding
- One-Hot Encoding
- Feature Scaling using StandardScaler
- Train-Test Split
- Validation Split

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Titanic-Survival-Prediction.git
```

Navigate to the project

```bash
cd Titanic-Survival-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# 💻 Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Pickle

---

# 📈 Prediction Workflow

```
User Input

↓

Label Encoding

↓

One Hot Encoding

↓

Feature Scaling

↓

Deep Learning Model

↓

Prediction Probability

↓

Survival Result
```

---

# 🎯 Example Prediction

Input

```
Passenger Class : 1

Gender : Female

Fare : 120

Embarked : Southampton

Parents/Children : 0

Siblings/Spouse : 1
```

Output

```
Survival Probability : 92.74%

Prediction

Passenger is likely to Survive
```

---

# 📊 Model Evaluation

Common evaluation metrics for binary classification include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

Example

```
Accuracy : 84%

Precision : 82%

Recall : 79%

ROC-AUC : 0.88
```

---

# 🚀 How to Improve Model Accuracy

Improving a Neural Network is not only about changing the model architecture. Better preprocessing, feature engineering, and hyperparameter tuning often have a greater impact.

## 1. Feature Scaling

Normalize numerical features using `StandardScaler` or `MinMaxScaler`.

Benefits:

- Faster convergence
- Stable gradients
- Better overall accuracy

---

## 2. Better Feature Engineering

Create new meaningful features such as:

- Family Size = SibSp + Parch + 1
- Is Alone
- Fare Per Person
- Passenger Title (Mr, Mrs, Miss, etc.)
- Age Group (if Age is available)

Good features often improve performance more than adding layers.

---

## 3. Handle Missing Values Properly

Instead of removing rows:

- Median Imputation
- Mean Imputation
- KNN Imputation

This helps preserve valuable training data.

---

## 4. Use Batch Normalization

Adding Batch Normalization after Dense layers helps by:

- Stabilizing training
- Reducing internal covariate shift
- Allowing faster learning

Example

```
Dense
↓

BatchNormalization
↓

ReLU
```

---

## 5. Add Dropout Layers

Dropout prevents overfitting by randomly disabling neurons during training.

Typical values:

```
0.2

0.3

0.5
```

---

## 6. Reduce Learning Rate

Instead of

```
learning_rate = 0.01
```

Try

```
0.001

or

0.0005
```

A smaller learning rate usually leads to smoother convergence.

---

## 7. Train for More Epochs

Instead of training for only a few epochs, use:

```
epochs = 100
```

combined with

```
EarlyStopping
```

to automatically stop when validation performance no longer improves.

---

## 8. Hyperparameter Tuning

Experiment with:

- Number of Hidden Layers
- Number of Neurons
- Learning Rate
- Batch Size
- Activation Functions
- Optimizers
- Dropout Rate

Tools such as **KerasTuner** can automate this process.

---

## 9. Handle Class Imbalance

If one class has significantly more samples than the other, consider:

- Class Weights
- SMOTE
- Random Over Sampling

This improves the model's ability to recognize the minority class.

---

## 10. Evaluate More Than Accuracy

Accuracy alone may be misleading.

Also evaluate:

- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

These metrics provide a more complete understanding of model performance.

---

## 11. Cross Validation

Use K-Fold Cross Validation to ensure the model generalizes well and does not overfit a particular train-test split.

---

## 12. Experiment with Different Architectures

Example:

```
Input

↓

Dense(64)

↓

BatchNormalization

↓

Dropout(0.3)

↓

Dense(32)

↓

BatchNormalization

↓

Dropout(0.3)

↓

Dense(16)

↓

Output
```

Sometimes a smaller, well-regularized network performs better than a larger one.

---

# 📌 Future Improvements

- Add Passenger Age
- Add Cabin Information
- SHAP Explainability
- Plotly Interactive Charts
- Docker Deployment
- CI/CD Pipeline
- FastAPI REST API
- Cloud Deployment (AWS/GCP/Azure)
- Model Monitoring
- Hyperparameter Optimization using KerasTuner

---

# 🤝 Contributing

Contributions are welcome!

Feel free to fork this repository, improve the project, and submit a pull request.

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Arun Jawlia**

Full Stack + AI Engineer

Passionate about Machine Learning, Deep Learning, Generative AI, and Production-Ready AI Applications.

If you found this project helpful, consider giving it a ⭐ on GitHub.