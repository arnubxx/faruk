#!/usr/bin/env python3
import os
from typing import Optional, Tuple

import numpy as np
import streamlit as st

# Optional: scikit-learn + joblib for local model
try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    joblib = None  # type: ignore
    TfidfVectorizer = object  # type: ignore


@st.cache_resource(show_spinner=False)
def load_models() -> Tuple[Optional[object], Optional[object], Optional[object]]:
    """Load SVM classifier, label encoder, and optional vectorizer.

    Returns: (svm_model, label_encoder, vectorizer)
    """
    svm_model = None
    label_encoder = None
    vectorizer = None

    if joblib is None:
        return svm_model, label_encoder, vectorizer

    try:
        if os.path.exists("svm_model.joblib"):
            svm_model = joblib.load("svm_model.joblib")
    except Exception as e:
        st.error(f"Failed to load svm_model.joblib: {e}")

    try:
        if os.path.exists("label_encoder.joblib"):
            label_encoder = joblib.load("label_encoder.joblib")
    except Exception as e:
        st.error(f"Failed to load label_encoder.joblib: {e}")

    # Attempt to load a saved vectorizer if present
    for name in ("vectorizer.joblib", "tfidf_vectorizer.joblib", "bow_vectorizer.joblib"):
        if os.path.exists(name):
            try:
                vectorizer = joblib.load(name)
                break
            except Exception as e:
                st.warning(f"Found {name} but failed to load: {e}")

    return svm_model, label_encoder, vectorizer


def app_4class_classifier():
    st.set_page_config(page_title="Sentence Classifier", layout="centered")
    st.title("Sentence Classifier (4 classes)")
    st.caption("Negative · Neutral · Positive · Suicidal")

    svm_model, label_encoder, vectorizer = load_models()

    # Validate models
    if svm_model is None or label_encoder is None:
        st.error("Model files not fully available. Ensure svm_model.joblib and label_encoder.joblib are present.")
        st.stop()

    # Vectorizer guidance
    if vectorizer is None:
        st.warning(
            "Vectorizer missing. Please add a saved TF-IDF vectorizer (vectorizer.joblib) used during training."
        )
        st.info(
            "Without the original vectorizer, predictions may be inaccurate."
        )

    text = st.text_area("Enter a sentence", height=140, placeholder="Type a sentence to classify...")
    if st.button("Classify") and text.strip():
        if vectorizer is None:
            st.error("Cannot classify without the saved vectorizer. Upload vectorizer.joblib to the repository.")
            return

        try:
            X = vectorizer.transform([text])
        except Exception as e:
            st.error(f"Vectorizer failed to transform input: {e}")
            return

        try:
            y_pred = svm_model.predict(X)
            # Convert numeric class to label
            label = label_encoder.inverse_transform(np.asarray(y_pred))
            st.metric("Prediction", label[0])

            # Optional probabilities if available
            if hasattr(svm_model, "predict_proba"):
                try:
                    probs = svm_model.predict_proba(X)[0]
                    classes = label_encoder.classes_
                    st.subheader("Class probabilities")
                    for c, p in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True):
                        st.write(f"{c}: {p:.4f}")
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Prediction failed: {e}")


def main():
    app_4class_classifier()


if __name__ == "__main__":
    main()
