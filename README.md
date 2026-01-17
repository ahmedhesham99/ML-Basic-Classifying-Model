# Advertising Click Prediction — Streamlit App

Project Summary: A machine learning model that takes datasets to predict whether a user will click on advertising ad or not, to increase conversion rate of next campaigns.




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


![Liveapp1](https://github.com/user-attachments/assets/b0c2ea6a-d359-4a36-be29-5b9fcbe7487b)
![Liveapp2](https://github.com/user-attachments/assets/5085e271-42f0-4ec8-8877-0d0131f0a080)
![Liveapp3](https://github.com/user-attachments/assets/11044f19-4805-49c7-bc4d-2fc49551424f)



