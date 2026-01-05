# HDF5 (.h5) Streamlit Viewer

A lightweight Streamlit app to explore HDF5 files: browse groups and datasets, view attributes, preview values, and download slices as `.npy` or `.csv`.

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

4. In the sidebar, either select `distilbert_model_weights.h5` (if present), upload a `.h5/.hdf5` file, or use "From URL" to fetch a file.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On https://streamlit.io/cloud, create a new app pointing to `streamlit_app.py`.
3. In the app, upload your `.h5` file via the sidebar or paste a direct URL. (Avoid committing very large weight files to the repo.)

### Large files

- Uploads are limited by Streamlit's `server.maxUploadSize` (set to 1024 MB in `.streamlit/config.toml`) and by platform limits. If uploads fail, prefer the "From URL" option to let the app download directly from object storage (S3, GCS) or a public link.

## Notes

- The preview limits rows/cols to keep the UI responsive. Use the download buttons for full arrays.
- If your file is a Keras model `.h5`, this app inspects datasets; it does not load the model with TensorFlow.