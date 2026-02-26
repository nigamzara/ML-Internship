import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None
    Pool = None

st.title("Mental Health Prediction")

# --- 6 Raw Features Input ---
Growing_Stress = st.selectbox("Growing Stress?", ["Yes", "No"])
Mood_Swings = st.selectbox("Mood Swings?", ["Yes", "No"])
Coping_Struggles = st.selectbox("Coping Struggles?", ["Yes", "No"])
Social_Weakness = st.selectbox("Social Weakness?", ["Yes", "No"])
Work_Interest = st.selectbox("Work Interest?", ["Yes", "No"])
Family_History = st.selectbox("Family History?", ["Yes", "No"])

# --- Use saved encoders/scaler/feature names when available ---
saved_objects = None
try:
    saved_objects = joblib.load('all_objects.joblib')
except Exception:
    # fall back to other files later
    saved_objects = None

# Default feature list (if not present in saved objects)
feature_names = ['Growing_Stress', 'Mood_Swings', 'Coping_Struggles', 'Social_Weakness', 'Work_Interest', 'family_history']
encoders = {}
scaler = None
if isinstance(saved_objects, dict):
    feature_names = saved_objects.get('feature_names', feature_names)
    encoders = saved_objects.get('encoders', {}) or {}
    scaler = saved_objects.get('scaler', None)

# Build raw input dict in the same naming as feature_names
raw_inputs = {
    'Growing_Stress': Growing_Stress,
    'Mood_Swings': Mood_Swings,
    'Coping_Struggles': Coping_Struggles,
    'Social_Weakness': Social_Weakness,
    'Work_Interest': Work_Interest,
    'family_history': Family_History,
}

# Map raw inputs through encoders (label encoders) when available
X_array = np.zeros((1, len(feature_names)))
for i, fname in enumerate(feature_names):
    val = raw_inputs.get(fname)
    if val is None:
        X_array[0, i] = 0
        continue
    encoder = encoders.get(fname)
    try:
        if encoder is not None:
            X_array[0, i] = int(encoder.transform([val])[0])
        else:
            # fallback: map Yes/No to 1/0
            X_array[0, i] = 1 if str(val).lower() in ('yes', 'y', 'true', '1') else 0
    except Exception:
        # encoder might expect different string format; try numeric cast
        try:
            X_array[0, i] = float(val)
        except Exception:
            X_array[0, i] = 0

# Apply scaler if available
X_for_model = X_array
if scaler is not None:
    try:
        X_for_model = scaler.transform(X_array)
    except Exception:
        X_for_model = X_array

def try_load_model():
    # Try common joblib files and return first usable model object
    # Prefer an explicit CatBoost model file if present
    cb_files = ["catboost_model.cbm", "catboost_model.model", "model.cbm", "model.model"]
    if (CatBoostClassifier is not None) and any(os.path.exists(p) for p in cb_files):
        for p in cb_files:
            if os.path.exists(p):
                try:
                    m = CatBoostClassifier()
                    m.load_model(p)
                    return m
                except Exception:
                    # try next
                    continue

    # If all_objects.joblib contains a catboost_model key, prefer it
    if os.path.exists('all_objects.joblib'):
        try:
            ao = joblib.load('all_objects.joblib')
            if isinstance(ao, dict):
                if 'catboost_model' in ao:
                    return ao['catboost_model']
                # some workflows keep the raw model under 'model'
                if hasattr(ao.get('model'), '__class__') and 'catboost' in type(ao.get('model')).__module__:
                    return ao.get('model')
        except Exception:
            pass

    # Fall back to earlier candidates
    candidates = ["all_objects.joblib", "model.joblib", "best_model.joblib"]
    for fname in candidates:
        if os.path.exists(fname):
            try:
                obj = joblib.load(fname)
            except Exception:
                continue
            # If a dict-like container, try to extract common keys
            if isinstance(obj, dict):
                for key in ("catboost_model", "model", "best_model"):
                    if key in obj:
                        return obj[key]
            return obj
    return None


model = try_load_model()
if model is None:
    st.error("Model not found in workspace (looked for joblib files).")
    st.stop()

# Determine positive-class index and default threshold for treatment decision
try:
    classes = list(getattr(model, 'classes_', []))
    positive_idx = next((i for i, c in enumerate(classes) if str(c) in ('1', '1.0', 'True', 'true', 'yes', 'Yes')), 1 if len(classes) > 1 else 0)
except Exception:
    positive_idx = 1

# fixed threshold used for treatment decision
threshold = 0.1334678679704666

# Prepare input for prediction; use CatBoost Pool only for CatBoost models
X_for_pred = X_for_model
is_catboost_model = False
try:
    modname = type(model).__module__ if model is not None else ''
    clsname = type(model).__name__ if model is not None else ''
    if ('catboost' in modname) or clsname.lower().startswith('catboost') or (
        CatBoostClassifier is not None and isinstance(model, (CatBoostClassifier, CatBoostRegressor))
    ):
        is_catboost_model = True
except Exception:
    is_catboost_model = False

if is_catboost_model and Pool is not None:
    try:
        X_for_pred = Pool(X_for_model, feature_names=feature_names)
    except Exception:
        X_for_pred = X_for_model

# If the model expects a different feature shape (e.g., 38 one-hot columns),
# build that vector from known mappings so we match the trained model input.
expected_n = None
try:
    expected_n = int(getattr(model, 'n_features_in_', None))
except Exception:
    expected_n = None
if expected_n is None:
    try:
        booster = getattr(model, 'get_booster', lambda: None)()
        if booster is not None and hasattr(booster, 'feature_names'):
            expected_n = len(booster.feature_names)
    except Exception:
        expected_n = None

if expected_n == 38:
    # original 38 training columns (from previous app version)
    model_columns = ['Gender_Male', 'Country_Belgium', 'Country_Bosnia and Herzegovina',
                     'Country_Brazil', 'Country_Canada', 'Country_Colombia', 'Country_Costa Rica',
                     'Country_Croatia', 'Country_Czech Republic', 'Country_Denmark', 'Country_Finland',
                     'Country_France', 'Country_Georgia', 'Country_Greece', 'Country_India',
                     'Country_Israel', 'Country_Italy', 'Country_Mexico', 'Country_Moldova',
                     'Country_Netherlands', 'Country_New Zealand', 'Country_Nigeria',
                     'Country_Philippines', 'Country_Poland', 'Country_Portugal', 'Country_Russia',
                     'Country_Singapore', 'Country_South Africa', 'Country_Sweden', 'Country_Thailand',
                     'Country_United States', 'self_employed_Yes', 'family_history_Yes',
                     'Growing_Stress_Yes', 'mental_health_interview_No', 'mental_health_interview_Yes',
                     'care_options_Not sure', 'care_options_Yes']

    X38 = np.zeros((1, len(model_columns)))
    # Map known raw inputs to the one-hot / *_Yes columns
    map_yes = {
        'Growing_Stress': 'Growing_Stress_Yes',
        'family_history': 'family_history_Yes',
        'Mood_Swings': 'Mood_Swings_Yes',
        'Coping_Struggles': 'Coping_Struggles_Yes',
        'Social_Weakness': 'Social_Weakness_Yes',
        'Work_Interest': 'Work_Interest_Yes',
    }
    for raw_name, col_name in map_yes.items():
        val = raw_inputs.get(raw_name)
        if val is None:
            continue
        if str(val).lower() in ('yes', 'y', 'true', '1') and col_name in model_columns:
            X38[0, model_columns.index(col_name)] = 1

    # If encoders had mapped some categorical values to one-hot columns, we can't reconstruct them here;
    # but setting the known *_Yes flags is likely sufficient to match training inputs for these fields.
    X_for_model = X38
    X_for_pred = X38

# Predict only when user clicks the Predict button
if st.button('Predict'):
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_for_pred)

            # determine positive-class index (prefer class '1')
            positive_idx = None
            try:
                classes = list(model.classes_)
                for i, c in enumerate(classes):
                    if str(c) in ('1', '1.0', 'True', 'true', 'yes', 'Yes'):
                        positive_idx = i
                        break
                if positive_idx is None:
                    positive_idx = 1 if len(classes) > 1 else 0
            except Exception:
                positive_idx = 1

            prob_pos = float(probs[0][positive_idx])
            # fixed threshold to decide treatment
            threshold = 0.1334678679704666
            treatment_needed = 1 if prob_pos >= threshold else 0

            st.write('Treatment needed:', treatment_needed)
            st.write('Probabilities:', {str(c): float(p) for c, p in zip(model.classes_, probs[0])})
        else:
            preds = model.predict(X_for_pred)
            st.write('Treatment needed:', int(preds[0]))
    except Exception as e:
        st.error(f'Prediction error: {e}')

# Analyze single-feature flips from current selections
if st.button('Analyze feature flips'):
    try:
        st.write('Testing single-feature toggles (Yes/No) for the six inputs...')
        rows = []
        for fname in list(raw_inputs.keys()):
            cur = raw_inputs.get(fname)
            toggled = raw_inputs.copy()
            toggled[fname] = 'No' if str(cur).lower() in ('yes','y','true','1') else 'Yes'

            # encode toggled inputs similar to main flow
            X_test = np.zeros((1, len(feature_names)))
            for i, fn in enumerate(feature_names):
                val = toggled.get(fn)
                if val is None:
                    X_test[0, i] = 0
                    continue
                enc = encoders.get(fn)
                try:
                    if enc is not None:
                        X_test[0, i] = int(enc.transform([val])[0])
                    else:
                        X_test[0, i] = 1 if str(val).lower() in ('yes','y','true','1') else 0
                except Exception:
                    try:
                        X_test[0, i] = float(val)
                    except Exception:
                        X_test[0, i] = 0

            if scaler is not None:
                try:
                    Xm = scaler.transform(X_test)
                except Exception:
                    Xm = X_test
            else:
                Xm = X_test

            Xp = Xm
            if expected_n == 38:
                X38 = np.zeros((1, len(model_columns)))
                for raw_name, col_name in {
                    'Growing_Stress': 'Growing_Stress_Yes',
                    'family_history': 'family_history_Yes',
                    'Mood_Swings': 'Mood_Swings_Yes',
                    'Coping_Struggles': 'Coping_Struggles_Yes',
                    'Social_Weakness': 'Social_Weakness_Yes',
                    'Work_Interest': 'Work_Interest_Yes',
                }.items():
                    val = toggled.get(raw_name)
                    if val and str(val).lower() in ('yes','y','true','1') and col_name in model_columns:
                        X38[0, model_columns.index(col_name)] = 1
                Xp = X38

            if is_catboost_model and Pool is not None:
                try:
                    Xp_pred = Pool(Xp, feature_names=feature_names if expected_n != 38 else model_columns)
                except Exception:
                    Xp_pred = Xp
            else:
                Xp_pred = Xp

            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(Xp_pred)
                prob_pos = float(probs[0][positive_idx])
                treatment = 1 if prob_pos >= threshold else 0
            else:
                pred = int(model.predict(Xp_pred)[0])
                prob_pos = None
                treatment = pred

            rows.append({'feature': fname, 'from': cur, 'to': toggled[fname], 'prob_pos': prob_pos, 'treatment': treatment})

        st.table(pd.DataFrame(rows))
    except Exception as e:
        st.error(f'Analyze error: {e}')

# (Removed: Find minimal flips feature as requested)

# Debugging info
if st.checkbox('Show debug info'):
    st.subheader('Debug')
    st.write('Feature names used:', feature_names)
    st.write('Raw input values:', raw_inputs)
    st.write('Encoded numeric vector (pre-scale):')
    st.write(X_array)
    st.write('Vector passed to model (post-scale if applied):')
    st.write(X_for_model)
    try:
        st.write('Model type:', type(model))
        if hasattr(model, 'classes_'):
            st.write('Model classes:', list(model.classes_))
        if hasattr(model, 'predict_proba'):
            st.write('Predict proba on input:', model.predict_proba(X_for_pred))
    except Exception as e:
        st.write('Debug read failed:', e)


if __name__ == '__main__':
    # quick local smoke test when running python app.py
    try:
        print('Model loaded:', type(model))
        sample_for_pred = X_for_model
        if is_catboost_model and Pool is not None:
            try:
                sample_for_pred = Pool(X_for_model, feature_names=feature_names)
            except Exception:
                sample_for_pred = X_for_model
        print('Sample prediction:', model.predict(sample_for_pred))
    except Exception as e:
        print('Local test failed:', e)


