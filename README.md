# Titanic Survival Predictor - ML AP!
A production-style machine learning API that predicts the survival likelihood of Titanic passengers based on demographic and travel data.
Built to demonstrate end-to-end ML engineering practices - from data preprocessing and experiment tracking to containerised API deployment.

# Links
- **Docker Hub:**

## Project Overview
A KNN classifier is trained on the Titanic dataset and served as a REST API using FastAPI. The project follows production ML engineering practices including experiment tracking, model versioning, and containerised deployment.

## Tech Stack
- **Python** - core language
- **Pandas** - data preprocessing
- **Scikit-learn** - KNN classifier training and evaluation
- **MLflow** - experiment tracking and model versioning
- **Fast API** - REST API serving
- **Docker** - containerisation and deployment

## How It Works
1. Data is preprocessed - missing values handled, categorical features encoded
2. A KNN classifier is trained across multiple values of k, with each runlogged in MLflow (parameters, metrics, model artefact)
3. The best model is saved and loaded into a FastAPI application
4. The API accepts passenger data and returns a survival prediction with probability scores

## API Endpoints

### `GET /`
Health check
```json
{ "status": "ok", "model": "KNN Titanic Classifier" }
```

### `POST /predict`
Returns survival prediction for a passenger.

**Request body:**
```json
{
  "Pclass": 3,
  "Sex": 1,
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Embarked": 2
}
```

**Response:**
```json
{
  "survived": false,
  "survival_probability": 0.28,
  "death_probability": 0.71
}
```

**Encoding reference:**
- `Sex`: 0 = female, 1 = male
- `Embarked`: 0 = Cherbourg, 1 = Queenstown, 2 = Southampton

## Run Locally

### Option A - Python
```bash
git clone https://github.com/lamifesa/titanic-survival-predictor
cd titanic-survival-predictor
venv/Scripts/activate
pip install -r requirements.txt

# Regenerate the model by running data-preprocessing.ipynb
# Then start the API
uvicorn main:app --reload
```
Open `http://localhost:8000/docs` for the interactive API documentation.

### Option B - Docker
```bash
docker pull lamifesa/titanic-survival-predictor
docker run -p 8000:8000 lamifesa/titanic-survival-predictor
```

## Experimental Tracking
MLflow is used to track training runs across different values of k.
To view the experiment dashboard:
```bash
mlflow ui
```
Then open `http://localhost:5000`

## Project Structure
├── data-preprocessing.ipynb# Data cleaning, encoding, MLflow training
├── main.py                    # FastAPI application
├── columns.json               # Column order for inference
├── Dockerfile                 # Container definition
├── .dockerignore
├── requirements.txt
└── README.md 

## Note on Model File
The trained model (`knn_model.pkl`) is not committed to this repository due to file size. To regenerate it, run all cells in `data-processing.ipynb`, The model will be saved automatically.

## Development Note
This project was built as a learning exercise in ML engineering. 
Development included guidance and code review from AI tools (Claude).
