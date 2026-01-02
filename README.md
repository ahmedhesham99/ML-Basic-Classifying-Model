# Advertising Click Prediction — Streamlit App

Quick steps to get the app running locally:

1. Ensure your repository contains `rf_model.pkl` and `class_marketing_advertising.csv` in the project root, or set environment variables:

	- `RF_MODEL_PATH` — path to your model file (defaults to `rf_model.pkl` in repo root)
	- `TRAIN_CSV` — path to training CSV (defaults to `class_marketing_advertising.csv`)
	- `RF_MODEL_URL` — optional HTTP URL to download the model at runtime if not present

2. Install dependencies (create a virtualenv first if desired):

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app locally:

```bash
streamlit run streamlit_app.py
```

Online Streamlit Link:
https://ml-basic-classifying-model.streamlit.app/



