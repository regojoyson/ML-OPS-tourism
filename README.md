# Wellness Tourism Package Purchase Prediction — MLOps Pipeline

An end-to-end MLOps pipeline that predicts whether a customer will purchase the **Wellness Tourism Package** offered by "Visit with Us," a travel company. The pipeline automates data registration, preprocessing, model training with experiment tracking, and deployment to a Streamlit app on Hugging Face Spaces — all triggered via GitHub Actions CI/CD.

---

## Business Context

"Visit with Us" introduced a new Wellness Tourism Package and needed a scalable, automated system to identify potential buyers before contacting them. The manual approach was inconsistent and error-prone. This project builds a predictive ML model and wraps it in a fully automated MLOps pipeline to improve marketing efficiency and decision-making.

---

## Project Structure

```
ML-OPS-tourism/
├── tourism_project/
│   ├── data/                        # Raw dataset (tourism.csv)
│   ├── model_building/
│   │   ├── data_register.py         # Upload dataset to HF Hub
│   │   ├── prep.py                  # Data cleaning, encoding, train/test split
│   │   └── train.py                 # Model training, tuning, MLflow tracking, HF upload
│   ├── deployment/
│   │   ├── app.py                   # Streamlit web app
│   │   ├── Dockerfile               # Docker container config for HF Spaces
│   │   └── requirements.txt         # App dependencies
│   ├── hosting/
│   │   └── hosting.py               # Upload deployment files to HF Space
│   └── requirements.txt             # Pipeline/CI dependencies
└── .github/
    └── workflows/
        └── pipeline.yml             # GitHub Actions CI/CD workflow
```

---

## Pipeline Overview

The GitHub Actions workflow has 4 sequential jobs triggered on every push to `main`:

```
register-dataset → data-prep → model-training → deploy-hosting
```

| Job | Script | Description |
|---|---|---|
| `register-dataset` | `data_register.py` | Creates HF dataset repo and uploads `tourism.csv` |
| `data-prep` | `prep.py` | Cleans data, encodes features, splits train/test, uploads splits to HF |
| `model-training` | `train.py` | Trains XGBoost with GridSearchCV, logs to MLflow, saves model to HF |
| `deploy-hosting` | `hosting.py` | Uploads Streamlit app files to Hugging Face Space |

---

## Dataset

The dataset (`tourism.csv`) contains customer and interaction data with the following features:

**Customer Details**
| Feature | Description |
|---|---|
| `CustomerID` | Unique customer identifier (dropped during preprocessing) |
| `ProdTaken` | Target variable — purchased package (1: Yes, 0: No) |
| `Age` | Age of the customer |
| `TypeofContact` | Contact method (Company Invited / Self Inquiry) |
| `CityTier` | City category by development level (Tier 1/2/3) |
| `Occupation` | Customer's occupation (Salaried, Freelancer, etc.) |
| `Gender` | Gender (Male, Female) |
| `MaritalStatus` | Marital status (Single, Married, Divorced) |
| `Designation` | Job designation |
| `MonthlyIncome` | Gross monthly income |
| `Passport` | Holds valid passport (1: Yes, 0: No) |
| `OwnCar` | Owns a car (1: Yes, 0: No) |
| `NumberOfTrips` | Average annual trips |
| `NumberOfPersonVisiting` | Number of people accompanying |
| `NumberOfChildrenVisiting` | Children below age 5 accompanying |
| `PreferredPropertyStar` | Preferred hotel star rating |

**Interaction Data**
| Feature | Description |
|---|---|
| `PitchSatisfactionScore` | Satisfaction score for sales pitch |
| `ProductPitched` | Type of product pitched |
| `NumberOfFollowups` | Follow-ups made by salesperson |
| `DurationOfPitch` | Duration of sales pitch |

---

## Model

- **Algorithm:** XGBoost Classifier (`XGBClassifier`)
- **Preprocessing:** `StandardScaler` for numeric features, `OneHotEncoder` for categorical features via `make_column_transformer`
- **Tuning:** `GridSearchCV` with 5-fold cross-validation
- **Class imbalance:** Handled via `scale_pos_weight`
- **Classification threshold:** 0.45 (tuned for recall on class 1)
- **Experiment tracking:** MLflow (local server on port 5000 during CI run)
- **Model registry:** Saved as `best_tourism_model_v1.joblib` and uploaded to `samuelrego/tourism_package_model` on Hugging Face Hub

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9 |
| ML Framework | XGBoost, scikit-learn |
| Experiment Tracking | MLflow |
| Data & Model Registry | Hugging Face Hub |
| Web App | Streamlit |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Hosting | Hugging Face Spaces (Docker SDK) |

---

## Prerequisites

### 1. Hugging Face Setup
1. Go to your HF Profile → **Access Tokens**
2. Create a new token: type **Write**, name it `MLOps_Token`
3. Copy the token

### 2. GitHub Secrets
1. Go to your GitHub repo → **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
   - Name: `HF_TOKEN`
   - Value: paste your Hugging Face token

### 3. GitHub Actions Workflow
1. Create `.github/workflows/pipeline.yml` in the repo
2. Paste the pipeline YAML (see `pipeline.yml`)

---

## Deployment

The Streamlit app is containerized with Docker and hosted on Hugging Face Spaces:

- **HF Space:** [samuelrego/tourism-package-prediction](https://huggingface.co/spaces/samuelrego/tourism-package-prediction)
- **HF Model:** [samuelrego/tourism_package_model](https://huggingface.co/samuelrego/tourism_package_model)
- **HF Dataset:** [samuelrego/mlops_dataset](https://huggingface.co/datasets/samuelrego/mlops_dataset)

The app allows users to input customer details and get a real-time prediction on whether the customer is likely to purchase the Wellness Tourism Package.

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/regojoyson/ML-OPS-tourism.git
cd ML-OPS-tourism

# Install dependencies
pip install -r tourism_project/requirements.txt

# Run the Streamlit app
streamlit run tourism_project/deployment/app.py
```

---

## GitHub Repository

[https://github.com/regojoyson/ML-OPS-tourism](https://github.com/regojoyson/ML-OPS-tourism)
