# OptiCrop: Smart Agricultural Production Optimization Engine

## Description

OptiCrop is a machine learning-based agricultural recommendation system designed to optimize crop production using environmental and soil parameters. The project integrates machine learning algorithms with a Flask web application to provide intelligent crop recommendations. It assists farmers, agricultural researchers, agribusiness companies, and policymakers in making data-driven farming decisions that improve productivity, sustainability, and resource utilization.

---

## Entity Relationship Diagram

The Entity Relationship Diagram illustrates the database structure by defining key entities and their relationships. It helps organize agricultural data efficiently and supports accurate prediction and optimization processes. (See the rendered diagram above.)

**Entities:** User, SoilData, Crop, Dataset, MLModel, Prediction, Report

**Primary Keys:** user_id, soil_id, crop_id, dataset_id, model_id, prediction_id, report_id

**Relationships:**
- User → SoilData: One-to-Many
- SoilData → Prediction: One-to-One
- Crop → Prediction: One-to-Many
- MLModel → Prediction: One-to-Many
- Dataset → MLModel: One-to-Many
- Prediction → Report: One-to-Many

**Foreign Keys:** SoilData.user_id → User; Prediction.soil_id → SoilData; Prediction.crop_id → Crop; Prediction.model_id → MLModel; Report.prediction_id → Prediction

---

## Pre-requisites

Tools and libraries used to develop OptiCrop:

| Tool / Library | Purpose | Link |
|---|---|---|
| Anaconda Navigator | Environment & package management for data science | https://www.anaconda.com/download |
| PyCharm | Python IDE for coding and debugging | https://www.jetbrains.com/pycharm/ |
| NumPy | Numerical computing, array operations | https://numpy.org/doc/stable/ |
| Pandas | Data manipulation and analysis | https://pandas.pydata.org/docs/ |
| Scikit-learn | ML algorithms (classification, regression, clustering) | https://scikit-learn.org/stable/ |
| Matplotlib | Static/animated visualizations | https://matplotlib.org/stable/ |
| Seaborn | Statistical graphics on top of Matplotlib | https://seaborn.pydata.org/ |
| Flask | Lightweight web framework for model deployment | https://flask.palletsprojects.com/ |

---

## Project Flow (Epics & Stories)

**Epic 1: Define Problem and Understanding**
1. Identify and define the agricultural problem the system solves.
2. Gather and analyze business requirements.
3. Conduct a literature survey on existing crop recommendation techniques.
4. Analyze the social and business impact of the solution.

**Epic 2: Data Collection and Analysis**
1. Download and collect the agricultural dataset.
2. Import required Python libraries.
3. Read and explore the dataset structure.
4. Univariate analysis of individual features.
5. Bivariate analysis of feature relationships.
6. Multivariate analysis across multiple variables.

**Epic 3: Data Pre-Processing**
1. Check for null values.
2. Detect and treat outliers.
3. Extract seasonal crop information.
4. Split data into training and testing sets.

**Epic 4: Model Building**
1. Apply K-Means Clustering.
2. Train a Logistic Regression model (plus KNN, Decision Tree, Random Forest for comparison).
3. Evaluate and compare model performance.
4. Save the best model and generate predictions.

**Epic 5: Application Building**
1. Design HTML pages for the interface.
2. Build the Flask backend integrated with the trained model.
3. Run, test, and validate the application.

---

## Epic 1: Define Problem and Understanding

### Specify the Business Problem

Modern agriculture faces challenges such as poor crop selection, inefficient resource utilization, changing environmental conditions, and reduced farming productivity. Farmers often struggle to determine the most suitable crop based on soil nutrients and climate conditions, leading to financial losses and low yield.

This phase analyzes key environmental and soil parameters — Nitrogen (N), Phosphorous (P), Potassium (K), temperature, humidity, pH, rainfall, and season — and their impact on crop growth. The goal is an intelligent, data-driven solution assisting farmers, researchers, agribusinesses, and policymakers in making better farming decisions through machine learning and predictive analytics.

### Business Requirements

- Accurately analyze agricultural/environmental data to generate intelligent crop recommendations.
- Process N, P, K, temperature, humidity, rainfall, pH, and seasonal conditions to identify the most suitable crop.
- Provide a simple, responsive, user-friendly web interface for real-time predictions.
- Support multiple ML algorithms: KNN, Logistic Regression, Decision Tree, Random Forest, K-Means Clustering.
- Include preprocessing, feature scaling, model evaluation, and prediction visualization.
- Optimize water, fertilizer, and soil nutrient usage while reducing crop failure risk.
- Maintain scalability and flexibility for future enhancements (advanced models, larger datasets, cloud deployment, smart farming integration).

### Literature Survey

This phase studies existing agricultural recommendation systems, crop prediction technologies, and smart farming solutions via research papers, journals, and case studies. It compares ML algorithms — Decision Trees, Random Forest, KNN, Logistic Regression, Neural Networks, and clustering methods — for crop recommendation, yield prediction, and environmental monitoring, along with preprocessing techniques, feature engineering, and evaluation metrics. This helps identify gaps and select suitable methodologies for OptiCrop's architecture and prediction optimization.

### Social and Business Impact

By providing accurate, data-driven crop recommendations, OptiCrop can help farmers improve productivity, reduce risk, and increase profitability through better crop selection. It promotes sustainable farming by optimizing water, fertilizer, and pesticide use, supporting precision agriculture. For agribusinesses, researchers, and policymakers, it offers predictive decision support for agricultural planning and resource allocation — contributing to rural development, food security, and the broader adoption of intelligent farming solutions.

---

## Epic 2: Data Collection and Analysis

### Download the Dataset

Dataset used: `Crop_recommendation.csv`-style data (representative synthetic dataset generated for this project, following the same structure as the Kaggle "Smart Agricultural Production Optimizing Engine" dataset — N, P, K, temperature, humidity, ph, rainfall, label). Reference: https://www.kaggle.com/datasets/chitrakumari25/smart-agricultural-production-optimizing-engine

### Importing the Libraries

NumPy and Pandas handle numerical computation and dataset structures; Matplotlib and Seaborn (fivethirtyeight style) handle visualization; Scikit-learn modules handle model training, clustering, splitting, and evaluation. See `notebooks/OptiCrop_Model_Building.ipynb`.

### Read the Dataset

The dataset is loaded with `pd.read_csv()`; `df.head()` displays the first records to confirm structure — N, P, K, temperature, humidity, ph, rainfall, and the crop `label`.

### Univariate Analysis

Distribution plots (`histplot`/`distplot`) and count plots examine each feature individually — N, P, K, temperature, humidity, pH, and rainfall — to understand agricultural data patterns for crop prediction.

### Bivariate Analysis

A scatter plot of humidity vs. crop label visualizes how environmental humidity relates to crop suitability across different crop types.

### Multivariate Analysis

Pairplots and correlation heatmaps across N, P, K, temperature, humidity, and rainfall reveal patterns and interactions among multiple environmental variables simultaneously, supporting more informed crop prediction.

---

## Epic 3: Data Pre-Processing

### Checking for Null Values

`df.shape`, `df.info()`, and `df.isnull().sum()` confirm dataset structure, data types, and absence of missing values — ensuring clean data for training.

### Handling Outliers

Boxplots visually reveal outliers. Bounds are computed via IQR:
- Upper Bound = Q3 + 1.5 × IQR
- Lower Bound = Q1 − 1.5 × IQR

The Potassium (K) feature shows outliers; a log transformation (`np.log1p`) is applied to normalize its distribution.

### Extracting Seasonal Crops

Crops are grouped by temperature, humidity, and rainfall ranges into Summer, Winter, and Rainy season buckets, enabling season-aware recommendations.

### Splitting Data into Train and Test Sets

`train_test_split()` divides features (X: N, P, K, temperature, humidity, ph, rainfall) and target (y: label) into training and testing sets (80/20, stratified), enabling reliable evaluation.

---

## Epic 4: Model Building

### K-Means Clustering

K-Means groups similar agricultural conditions based on environmental features. The Elbow Method (plotting WCSS vs. number of clusters) identifies the optimal cluster count — the "elbow point" where the rate of decrease sharply changes. The model is trained using `fit()`/`fit_predict()`.

### Logistic Regression

Logistic Regression is trained on the scaled features using `fit()`, with predictions generated via `predict()`. Performance is evaluated for accuracy and recommendation reliability. KNN, Decision Tree, and Random Forest are trained in parallel for comparison, per the project's business requirements.

### Evaluating Model Performance and Saving the Best Model

All models are compared using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix. In this project, **Random Forest** performed best (~93% accuracy on held-out test data). The finalized model is serialized with `pickle.dump()` and stored as `model/model.pkl` (with `model/scaler.pkl` for feature scaling) for reuse in the Flask app without retraining.

### Predict the Best Crop Based on Given Parameters

The saved model predicts the most suitable crop from user-provided N, P, K, temperature, humidity, ph, and rainfall values via `model.predict()`, helping farmers make data-driven cultivation decisions.

---

## Epic 5: Application Building

### Building HTML Pages

The web app has three pages, linked via a top navigation bar:
- **Home** — introduces OptiCrop and its purpose.
- **About** — explains the ML-driven approach and objectives.
- **Find Your Crop** — a form collecting N, P, K, temperature, humidity, pH, and rainfall, submitted to the backend for prediction.

### Build Python Backend Code

`app.py` (Flask):
- Loads `model.pkl` and `scaler.pkl` using `pickle`.
- Defines routes for Home, About, and Find Your Crop (`@app.route`) rendering their templates.
- On POST to `/findyourcrop`, collects form inputs, converts them to numeric values, scales them, and calls `model.predict()` to generate the crop recommendation, displayed back on the same page.
- `app.run(debug=True)` starts the Flask development server.

### Run the Application

Open the project in VS Code (or PyCharm), ensure the virtual environment is active with all dependencies installed, then run:
```
python app.py
```
Click the local server URL shown in the terminal (typically `http://127.0.0.1:5000`) to open the app in a browser and test the Home, About, and Find Your Crop pages.

---

## Conclusion

The OptiCrop project successfully demonstrates how machine learning can be applied to solve a real agricultural problem: helping farmers choose the most suitable crop based on soil nutrients and environmental conditions. Through systematic data collection, exploratory analysis, preprocessing, and comparison of multiple algorithms (KNN, Logistic Regression, Decision Tree, Random Forest, K-Means), a Random Forest model was selected and deployed via a Flask web application with an interactive, user-friendly interface.

The project delivers a working end-to-end pipeline — from raw environmental data to a real-time crop recommendation accessible through a browser — and lays the foundation for future enhancements such as larger real-world datasets, additional environmental factors, yield-quantity prediction, cloud deployment, and integration with IoT-based soil sensors for fully automated smart farming.
