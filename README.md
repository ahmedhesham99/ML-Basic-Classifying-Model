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

Notes for deploying to Streamlit Cloud:

- Do not hard-code absolute Windows paths in the app — use the defaults above or set the `RF_MODEL_PATH` env var in the Streamlit app settings.
- If the model file is large (>100 MB), add it via Git LFS or host it at `RF_MODEL_URL` and set that env var in Streamlit so the app can download it at startup.

If you want, I can prepare a short deployment checklist for Streamlit Cloud and update the repo automated settings.

Upload a CSV with the same columns as the training data (or use the single-sample form). If you prefer, push the folder to GitHub and share the repository; visitors can run the two commands above.
