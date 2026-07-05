## Tech Stack

- **Language:** Python 3.10+
- **Backend:** Flask
- **ML Libraries:** scikit-learn (KNN, Logistic Regression, Decision Tree, Random Forest, K-Means)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Pickle
- **Frontend:** HTML5, CSS3
- **Dev Tools:** Jupyter Notebook, Git/GitHub

## Features

- Crop recommendation from 7 soil/environmental parameters
- Comparison of 4 ML algorithms with accuracy/precision/recall/F1 evaluation
- Best model (Random Forest) persisted for instant reuse
- Simple, responsive web interface (Home, About, Find Your Crop)
- Seasonal crop grouping (Summer/Winter/Rainy)

## Setup & Run Instructions

```bash
# 1. Clone the repository
git clone https://github.com/v-arun-kumar-03/OptiCrop.git
cd OptiCrop

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (generates model.pkl and scaler.pkl)
python train_model.py

# 5. Run the application
python app.py

# 6. Open in browser
http://127.0.0.1:5000
```

## Dataset

A representative crop-recommendation dataset (2,200 records, 22 crop classes) structured with columns: `N, P, K, temperature, humidity, ph, rainfall, label`, modeled on the Kaggle "Smart Agricultural Production Optimizing Engine" dataset.

## Known Limitations

- Flask's development server is single-threaded (not production-scale)
- No authentication (public, stateless prediction tool)
- Dataset is representative/synthetic, not large-scale real-world sensor data

## Future Scope

- Live weather API integration for real-time rainfall/temperature input
- Researcher/policymaker dashboard for regional trend analysis
- Cloud deployment with a production WSGI server

## License

This project is for educational purposes as part of the SmartBridge / SkillWallet AI-ML-and-GEN-AI Track.
