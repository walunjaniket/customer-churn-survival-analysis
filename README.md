# customer-churn-survival-analysis
Customer Churn Survival Analysis using Cox Proportional Hazards model and Kaplan-Meier curves

---

Download pretrained model from Releases (v1.0)

---

## Problem Statement
Customer churn — when customers stop doing business with a company — is one of the biggest challenges for telecom service providers.  
Instead of simply predicting whether a customer will churn, this project aims to **estimate how long a customer will stay** (customer lifetime) using **Survival Analysis**.

By modeling customer retention as a time-to-event problem, telecom companies can identify high-risk customers early and take proactive steps to reduce churn.

---

## Dataset
**Dataset:** Telco Customer Churn (Public dataset from IBM)

| Feature | Description |
|----------|--------------|
| `customerID` | Unique customer identifier |
| `gender` | Male / Female |
| `SeniorCitizen` | 1 = Yes, 0 = No |
| `Partner` | Has partner or not |
| `Dependents` | Has dependents or not |
| `tenure` | Number of months customer has stayed |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Total amount billed |
| `Contract` | Contract type: Month-to-month / One year / Two year |
| `PaymentMethod` | Payment type (e.g., Bank Transfer, Electronic Check) |
| `Churn` | Yes / No (used as event variable) |

- **Duration variable:** `tenure`  
- **Event variable:** `churn`  

---

## Methodology

### 1. Data Preprocessing & EDA
- Cleaned and standardized column names  
- Converted data types (especially `TotalCharges` → float)  
- Checked missing values and outliers  
- Explored churn patterns using Seaborn visualizations  

### 2. Feature Engineering
- Created an engineered feature: `autopayment`  
  → 1 if the payment method includes “automatic”, else 0  
- One-hot encoded categorical variables  
- Removed irrelevant or redundant columns  

### 3. Modeling — Cox Proportional Hazards (CPH)
- Used **Cox PH model** from `scikit-survival` to estimate hazard (risk of churn)  
- Evaluated using **Concordance Index (C-index)**  
- Saved trained model as `project 2 cph_model.pkl`  

### 4. Deployment — Interactive Streamlit Dashboard
Built a **Streamlit web application** that:
- Lets users enter a **new customer profile**
- Predicts:
  - **Churn Risk Group (Low / Medium / High)**  
  - **Risk Score**  
  - **Median Lifetime (months)**  
  - **24-month Survival Probability**
- Displays interactive:
  - Survival curves  
  - Model summary & top risk factors  
  - Global Kaplan-Meier plots  

---

## Key Insights
- Customers with **month-to-month contracts** have the **highest churn risk**.  
- **Automatic payments** strongly correlate with **longer survival**.  
- **Electronic check** users churn significantly faster.  
- **Senior citizens** show shorter retention periods on average.  

---

## Dashboard Overview

### Tabs
1. **Model Overview**  
   View Cox model summary, coefficients, and top churn risk drivers.
2. **Customer Prediction**  
   Input a new customer’s details to predict churn risk and visualize their survival curve.
3. **Global Insights**  
   Explore overall survival and contract-based retention patterns using Kaplan-Meier curves.

### Screenshots
*(Add your screenshots in a folder named `screenshots` and update the paths below)*

---

## Tech Stack
- Python (pandas, NumPy, matplotlib, seaborn, scikit-survival)
- Jupyter Notebook
- Plotly / Dash (for interactive dashboard)
- Parquet for data storage

---

## Files in Repository
- `Project 2 - Customer Churn Survival Analysis.ipynb` — main project notebook  
- `EDA.ipynb` — exploratory data analysis  
- `project 2 cph_model.pkl` — trained Cox Proportional Hazards model  
- `project 2 df_train.parquet` / `project 2 df_test.parquet` — processed datasets  
- `requirements.txt` — required Python packages  
- `Screenshots/` — dashboard visuals and results  
- `screen_recording.mp4` — short demo of dashboard

Demo Video

A short walkthrough of the Streamlit dashboard is available in
Project Resources/screen_recording.mp4

---

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/walunjaniket/customer-churn-survival-analysis.git

2. CNavigate to the project directory:
   ```bash
   cd customer-churn-survival-analysis

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Open Jupyter Notebook:
   ```bash
   jupyter notebook

5. Run Project 2 - Customer Churn Survival Analysis.ipynb to reproduce the results.

6. (Optional) Launch the interactive dashboard from the last notebook cell.
   ```bash
   git clone https://github.com/walunjaniket/customer-churn-survival-analysis.git

---

Download pretrained model from Releases (v1.0)

---

```markdown
## 👤 Author
**Aniket Walunj**  
Data Science Enthusiast | Machine Learning & Survival Analysis Projects  
[LinkedIn](https://www.linkedin.com/in/aniket-walunj-93a07864/) | [GitHub](https://github.com/walunjaniket)

```markdown
![Dashboard Overview](screenshots/dashboard_overview.jpg)
![Customer Prediction](screenshots/customer_prediction.jpg)
![Global Insights](screenshots/global_insights.jpg)
