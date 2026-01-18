# Forest Fire Risk Classification 🌲🔥

An educational AI-assisted project to predict forest fire risks using Logistic Regression.

## 📌 Problem Statement
Forest fires cause significant environmental and economic damage. Early detection and risk assessment can help in prevention. This project builds a simple classification model to predict whether a set of environmental conditions corresponds to a "Fire" or "Not Fire" scenario.

## 📂 Dataset
- **Name**: Algerian Forest Fires Dataset (UCI Machine Learning Repository)
- **Features Used**:
  - `Temperature`: Max temperature (°C)
  - `RH`: Relative Humidity (%)
  - `Ws`: Wind Speed (km/h)
  - `Rain`: Rainfall (mm)
- The dataset contains data from two regions (Bejaia and Sidi-Bel Abbes) concatenated. We merged them for this analysis.

## 🤖 Model Choice
- **Algorithm**: Logistic Regression
- **Why?**:
  - Simple, interpretable, and effective for binary classification.
  - Good for educational purposes (easy to understand weights/coefficients).
  - Fast training and inference.
- **Performance**: ~76% Accuracy on the test set.

## ⚠️ Limitations
- **Dataset Size**: The dataset is relatively small (~244 records).
- **Simplicity**: We count rainfall and other factors simply; real fire dynamics are complex (involving fuel moisture codes FFMC, DMC, etc., which we omitted for simplicity in inputs).
- **Educational Only**: This tool is for learning purposes and **NOT for real-world emergency use**.

## 🛠️ Tech Stack
- **Python**: Core language
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine Learning
- **Streamlit**: Web Interface

## 🤖 AI Assistant Role
This project was built with the help of an AI coding assistant (acting as a "Copilot").
- **Code Generation**: The AI generated the training script and Streamlit app.
- **Data Cleaning**: The AI handled the complexity of the dataset (header in the middle of the file).
- **Documentation**: The AI drafted this README.

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**:
   ```bash
   python train_model.py
   ```
   *This will create `model.pkl` and `scaler.pkl`.*

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## 📝 License
MIT License - Educational Use
