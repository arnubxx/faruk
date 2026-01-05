# 4-Class Sentence Classifier (Streamlit)

A Streamlit app that classifies a sentence into four classes:

- Negative
- Neutral
- Positive
- Suicidal

It loads a saved scikit-learn SVM (`svm_model.joblib`) and a `LabelEncoder` (`label_encoder.joblib`).
For accurate predictions, it also needs the TF-IDF vectorizer used during training
saved as `vectorizer.joblib`.

## Quick start (local)

1. Create/activate a Python 3.10+ environment.
2. Install deps:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run streamlit_app.py
```

4. Enter/paste text and click "Classify".

### Required files

Place these files in the project root:

- `svm_model.joblib` – trained `sklearn.svm.SVC` classifier
- `label_encoder.joblib` – label encoder with classes `['Negative', 'Neutral', 'Positive', 'Suicidal']`
- `vectorizer.joblib` – TF-IDF vectorizer used during training (must match training vocabulary)

Without `vectorizer.joblib`, the app cannot transform text to the expected feature space (`n_features_in_` of the SVM).

## Deploy to Render

This repo includes `render.yaml`. Deploy steps:

1. Push this folder to a GitHub repo.
2. On https://render.com, create a new Web Service and point it to your repo.
3. Render detects `render.yaml` and sets up the service.
4. The app will start with:

```bash
streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

Ensure the three model files listed above are committed to the repo.

1. Push this folder to a GitHub repo.
2. On https://streamlit.io/cloud, create a new app pointing to `streamlit_app.py`.
3. In the app, paste text and click "Classify". The model downloads on first run.

### Notes

- The SVM expects a feature dimension equal to the training vectorizer's vocabulary size. If you see errors about transformation, verify `vectorizer.joblib` is present and compatible.
- If you also have a Keras model (e.g., `distilbert_model_weights.h5`), it is not used by the current app. We can add a Keras/TensorFlow inference path on request.

## Development

- Python: 3.12 (venv)
- Install deps: `pip install -r requirements.txt`
- Run: `streamlit run streamlit_app.py`