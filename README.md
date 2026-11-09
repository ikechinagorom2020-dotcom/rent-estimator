# Rent Price Estimator & Negotiation Assistant

A learning project. You describe a rental flat and its asking rent. A machine learning model estimates a rough rent and a likely range, and an AI assistant turns those numbers into short, polite negotiation tips.

## How it works
1. A gradient boosting model (scikit-learn) was trained on about 4,650 rental listings from India in 2022.
2. The estimate, the range and the suggested offers are calculated by code, not written by an AI.
3. An AI assistant writes the negotiation tips from those numbers. A fixed note about the data and its error is added by the app.
4. The interface is built with Streamlit.

## Results
- Average error on 931 unseen listings: about 8,900 rupees (R2 0.78)
- Average error in 5-round cross-validation: about 10,500 rupees
- Typical error for a listing: about 20 percent

## Limits
- Data: House Rent Prediction Dataset from Kaggle (India, April to July 2022). Prices are in rupees and may be out of date.
- The model does not know a flat's condition, view or exact street.
- The estimate is a rough guide, not a market price or a professional valuation.
- The advice is written by an AI and is not legal or financial advice.

## Running it yourself
pip install -r requirements.txt
streamlit run app.py
Set GROQ_API_KEY as an environment variable or a Streamlit secret. Without it, the estimate still works but the advice is switched off.

Built with the help of an AI assistant.
