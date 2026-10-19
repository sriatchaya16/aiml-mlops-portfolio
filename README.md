# Machine Failure Prediction using MLOps

## Project Overview
This project predicts possible machine failures using machine learning. It provides a simple web application where users can enter machine operating conditions and receive a prediction.

**Note:** This is a learning project using a small synthetic dataset. Predictions are not suitable for real industrial safety or maintenance decisions.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- MLflow
- DVC
- Git and GitHub

## Project Structure
```text
aiml-mlops-portfolio/
├── app.py
├── data/
│   ├── machine_data.csv.dvc
│   └── .gitignore
├── models/
│   └── machine_failure_model.pkl
├── src/
│   └── train.py
├── .dvc/
├── .gitignore
└── README.md
```

## Features
- Predicts possible machine failure from operating measurements.
- Uses a Scikit-learn machine learning pipeline.
- Provides an interactive Streamlit interface.
- Tracks dataset versions with DVC.
- Logs model experiment metrics using MLflow.

## Dataset
The example dataset contains these features:
- Temperature
- Pressure
- Vibration
- Humidity
- Power Consumption

The target column is `Machine_Failure`, where `0` represents no failure and `1` represents possible failure.

## Installation and Usage

### 1. Clone the repository
```bash
git clone https://github.com/sriatchaya16/aiml-mlops-portfolio.git
cd aiml-mlops-portfolio
```

### 2. Create and activate a virtual environment
On Windows:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
python -m pip install streamlit pandas numpy scikit-learn mlflow dvc joblib
```

### 4. Retrieve the dataset using DVC
```bash
dvc pull
```
If no remote storage is configured, restore the dataset from the original local copy or configure a DVC remote first.

### 5. Train the model
```bash
python src/train.py
```

### 6. Run the web application
```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal.

## Experiment Tracking
MLflow is used to log model evaluation metrics, including accuracy, precision, recall, and F1-score.

## Future Improvements
- Use a larger, real-world machine failure dataset.
- Evaluate class imbalance and validate performance on unseen data.
- Configure remote DVC storage.
- Deploy the Streamlit application to a cloud platform.
- Explore AWS for cloud deployment and storage.

