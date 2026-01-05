# Hate Speech Classifier (Streamlit)

A minimal Streamlit app to classify text for hate speech using a Hugging Face model.

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

4. Enter/paste text and click "Classify". The app autoloads `cardiffnlp/twitter-roberta-base-hate` on first run.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On https://streamlit.io/cloud, create a new app pointing to `streamlit_app.py`.
3. In the app, paste text and click "Classify". The model downloads on first run.

### Large files

- Uploads are limited by Streamlit's `server.maxUploadSize` (set to 1024 MB in `.streamlit/config.toml`) and by platform limits. If uploads fail, prefer the "From URL" option to let the app download directly from object storage (S3, GCS) or a public link.

## Notes

- The preview limits rows/cols to keep the UI responsive. Use the download buttons for full arrays.
- If your file is a Keras model `.h5`, this app inspects datasets; it does not load the model with TensorFlow.
 - The classifier uses Hugging Face `transformers` (PyTorch). Initial model download occurs on first run.
 - If you want to switch models or add batch processing, we can extend the UI.