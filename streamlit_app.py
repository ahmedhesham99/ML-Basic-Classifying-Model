import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Use relative paths by default; allow overriding with environment variables
MODEL_PATH = os.getenv('RF_MODEL_PATH', 'rf_model.pkl')
TRAIN_CSV = os.getenv('TRAIN_CSV', 'class_marketing_advertising.csv')
MODEL_URL = os.getenv('RF_MODEL_URL', None)


@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        # attempt to download if MODEL_URL provided
        if MODEL_URL:
            try:
                st.info('Downloading model from RF_MODEL_URL...')
                urllib.request.urlretrieve(MODEL_URL, path)
                st.success('Model downloaded.')
            except Exception as e:
                st.error(f'Model not found at {path} and download failed: {e}')
                st.info('Set `RF_MODEL_PATH` or add the model file to the repository root.')
                return None
        else:
            st.error(f'Model not found at {path}')
            st.info('Either commit `rf_model.pkl` to the repo root, or set the env var `RF_MODEL_PATH`.')
            return None

    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f'Error loading model: {e}')
        return None


@st.cache_resource
def fit_transformers(train_csv=TRAIN_CSV):
    if not os.path.exists(train_csv):
        return None
    df = pd.read_csv(train_csv)
    # create time features if present
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Month'] = df['Timestamp'].dt.month
        df['Hour'] = df['Timestamp'].dt.hour
        df['Day'] = df['Timestamp'].dt.day

    # Fit label encoders for categorical columns used in notebook
    encoders = {}
    for col in ['Country', 'City', 'Ad Topic Line']:
        if col in df.columns:
            le = LabelEncoder()
            # fill NA with string to keep consistent mapping
            df[col] = df[col].fillna('')
            le.fit(df[col].astype(str))
            encoders[col] = le

    # Fit scaler for numeric columns
    numeric_cols = ['Age', 'Area Income', 'Daily Internet Usage', 'Daily Time Spent on Site', 'Month', 'Hour', 'Day']
    present_num = [c for c in numeric_cols if c in df.columns]
    scaler = None
    if present_num:
        scaler = StandardScaler()
        scaler.fit(df[present_num].fillna(0))

    return {'encoders': encoders, 'scaler': scaler, 'numeric_cols': present_num}


def apply_feature_engineering(df, transformers):
    df = df.copy()
    # time features
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Month'] = df['Timestamp'].dt.month
        df['Hour'] = df['Timestamp'].dt.hour
        df['Day'] = df['Timestamp'].dt.day

    encoders = transformers.get('encoders', {})
    # map categorical via fitted encoders, unseen -> -1
    for col, le in encoders.items():
        out_col = col.replace(' ', '_') + '_Enc' if ' ' in col else col + '_Enc'
        mapping = {v: i for i, v in enumerate(le.classes_)}
        df[col] = df[col].fillna('').astype(str)
        df[out_col] = df[col].map(lambda v: mapping.get(v, -1))

    # scale numeric features
    scaler = transformers.get('scaler')
    num_cols = transformers.get('numeric_cols', [])
    if scaler is not None and num_cols:
        vals = df[num_cols].fillna(0)
        scaled = scaler.transform(vals)
        for i, c in enumerate(num_cols):
            df[c + '_Sc'] = scaled[:, i]

    return df


def predict_df(model, transformers, df):
    df_fe = apply_feature_engineering(df, transformers)

    # Determine required feature names from the trained model when available
    required = None
    if hasattr(model, 'feature_names_in_'):
        required = list(model.feature_names_in_)
    elif hasattr(model, 'n_features_in_') and transformers and 'numeric_cols' in transformers:
        # Fallback: attempt to build features from typical engineered columns
        required = [
            'Country_Enc', 'City_Enc', 'Ad_Topic_Enc',
            'Age_Sc', 'Area Income_Sc', 'Daily Internet Usage_Sc', 'Daily Time Spent on Site_Sc',
            'Month_Sc', 'Hour_Sc', 'Day_Sc'
        ]

    if required is None:
        st.error('Could not determine required feature names from the model.')
        return None

    # Ensure all required columns exist in df_fe; fill missing with zeros
    for col in required:
        if col not in df_fe.columns:
            df_fe[col] = 0

    X = df_fe[required]
    try:
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    except Exception as e:
        st.error(f'Error during prediction: {e}')
        return None

    out = df.copy()
    # Map numeric predictions to friendly text for end users
    out['prediction'] = np.where(preds == 1, 'Prediction: Will click', 'Prediction: Will not click')
    if probs is not None:
        out['probability'] = probs
    return out


def main():
    st.title('Advertising Click Prediction')
    st.write('Using your model at the provided path and reproducing notebook feature engineering.')

    model = load_model()
    if model is None:
        return

    transformers = fit_transformers()
    if transformers is None:
        st.warning('Training CSV not found; predictions may fail if encoders/scaler are unavailable.')

    uploaded = st.file_uploader('Upload CSV for batch prediction', type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        result = predict_df(model, transformers, df)
        if result is not None:
            st.dataframe(result.head(50))
            csv = result.to_csv(index=False)
            st.download_button('Download predictions CSV', csv, 'predictions.csv', 'text/csv')

    st.markdown('---')
    st.header('Single sample prediction')
    with st.form('single'):
        country = st.text_input('Country', value='United States')
        city = st.text_input('City', value='New York')
        ad_topic = st.text_input('Ad Topic Line', value='Special Offer')
        age = st.number_input('Age', min_value=0.0, value=35.0)
        area_income = st.number_input('Area Income', min_value=0.0, value=50000.0)
        daily_internet = st.number_input('Daily Internet Usage', min_value=0.0, value=150.0)
        daily_time = st.number_input('Daily Time Spent on Site', min_value=0.0, value=65.0)
        month = st.number_input('Month', min_value=1, max_value=12, value=6)
        hour = st.number_input('Hour', min_value=0, max_value=23, value=12)
        day = st.number_input('Day', min_value=1, max_value=31, value=15)
        submitted = st.form_submit_button('Predict')

    if submitted:
        sample = pd.DataFrame([{
            'Country': country,
            'City': city,
            'Ad Topic Line': ad_topic,
            'Age': age,
            'Area Income': area_income,
            'Daily Internet Usage': daily_internet,
            'Daily Time Spent on Site': daily_time,
            'Month': month,
            'Hour': hour,
            'Day': day
        }])

        res = predict_df(model, transformers, sample)
        if res is not None:
            pred_text = res.iloc[0]['prediction']
            prob = res.iloc[0]['probability'] if 'probability' in res.columns else None
            if prob is not None:
                st.success(f"{pred_text} with probability {prob:.3f}")
            else:
                st.success(pred_text)


if __name__ == '__main__':
    main()
