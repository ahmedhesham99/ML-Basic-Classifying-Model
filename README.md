# Advertising Click Prediction — Streamlit App

Quick steps to get the app running locally:

1. Train and save the model (creates `rf_model.pkl`):

```bash
python export_model.py
```

2. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Upload a CSV with the same columns as the training data (or use the single-sample form). If you prefer, push the folder to GitHub and share the repository; visitors can run the two commands above.
