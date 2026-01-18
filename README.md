# 🌲🔥 Forest Fire Risk Classification

An **educational AI-assisted machine learning project** that predicts forest fire risk based on environmental conditions using Logistic Regression.

---

## 📌 Problem Statement

Forest fires cause significant environmental, economic, and ecological damage. Early identification of high-risk conditions can help authorities and communities take preventive actions.

This project demonstrates a **prototype risk classification system** that predicts whether given environmental conditions correspond to a **Fire** or **Not Fire** scenario.

---

## 📂 Dataset

* **Name:** Algerian Forest Fires Dataset
* **Source:** UCI Machine Learning Repository

### Features Used

* **Temperature (°C):** Daily maximum temperature
* **RH (%):** Relative humidity
* **Ws (km/h):** Wind speed
* **Rain (mm):** Rainfall

The dataset contains records from two regions (Bejaia and Sidi-Bel Abbes). Both regions were merged and cleaned to form a single dataset for training and evaluation.

---

## 🤖 Model Choice

* **Algorithm:** Logistic Regression

### Why Logistic Regression?

* Simple and interpretable binary classification model
* Suitable for educational and prototype systems
* Fast training and inference
* Easy to explain in academic evaluations

**Model Performance:** ~76% accuracy on the test dataset.

---

## ⚠️ Limitations

* **Dataset Size:** Relatively small (~244 records)
* **Feature Scope:** Real wildfire behavior depends on complex factors (fuel moisture indices like FFMC, DMC, etc.), which were excluded to keep inputs simple
* **No Real-Time Data:** Uses historical data only
* **Educational Purpose Only:** This system is a learning prototype and **must not be used for real-world emergency or safety decisions**

---

## 🛠️ Tech Stack

* **Python** – Core programming language
* **Pandas & NumPy** – Data processing
* **Scikit-learn** – Machine learning
* **Streamlit** – Interactive web interface

---

## 🤖 Role of AI Coding Assistant (Copilot)

This project was developed with the assistance of an **AI coding assistant (GitHub Copilot–style)**.

The AI assistant was used to:

* Suggest boilerplate code for data preprocessing and model training
* Assist in refactoring and organizing Python functions
* Speed up development of the Streamlit user interface
* Support debugging and code cleanup

All AI-generated suggestions were **reviewed, modified, and validated by the developer** before final integration.

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model

```bash
python train_model.py
```

This will generate:

* `model.pkl`
* `scaler.pkl`

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🟢 Final Note (For Mentors / Reviewers)

This project demonstrates the **workflow of building an AI-assisted classification system**, focusing on data preprocessing, model training, evaluation, and responsible AI usage rather than real-world deployment.
