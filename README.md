# Universal Bank — Personal Loan Propensity (Streamlit)

This Streamlit Cloud app lets you:
- Explore **5 actionable insights** with rich charts.
- Train and compare **Decision Tree**, **Random Forest**, and **Gradient Boosted Tree** using **5-fold CV** (AUC scoring).
- Upload a **new dataset** and download predictions for **Personal Loan** propensity.

## How to deploy on Streamlit Cloud
1. Create a new GitHub repo and upload these **flat files** (no folders):
   - `app.py`
   - `requirements.txt`
   - `README.md`
2. On Streamlit Cloud, create a new app and point it to `app.py`.
3. (Optional) Upload your `UniversalBank.csv` via the app sidebar, or use the synthetic sample provided by default.

## Data schema (expected columns)
`ID, Personal Loan, Age, Experience, Income, Zip code, Family, CCAvg, Education, Mortgage, Securities, CDAccount, Online, CreditCard`

- `Personal Loan`: 1 if accepted, 0 otherwise.
- `Education`: 1=Undergrad, 2=Graduate, 3=Advanced/Professional.

## Notes
- No version pins in `requirements.txt` (as requested).
- All plots are done with Plotly/Matplotlib embedded in Streamlit.
