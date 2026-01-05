#!/usr/bin/env python3
import io
import os
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import streamlit as st
import h5py
import zipfile
import requests

# Try to reuse summarize helper if present
try:
    from inspect_h5 import summarize_h5  # type: ignore
except Exception:
    summarize_h5 = None  # Fallback if not available


def list_datasets(h5: h5py.File) -> List[str]:
    paths: List[str] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    h5.visititems(visitor)
    paths.sort()
    return paths


def get_object_attrs(obj: h5py.Dataset | h5py.Group) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for k, v in obj.attrs.items():
        try:
            # Convert bytes to str if needed
            if isinstance(v, bytes):
                attrs[k] = v.decode('utf-8', errors='replace')
            elif isinstance(v, np.ndarray) and v.dtype.type is np.bytes_:
                attrs[k] = [x.decode('utf-8', errors='replace') for x in v.tolist()]
            else:
                attrs[k] = np.asarray(v).tolist() if isinstance(v, np.ndarray) else v
        except Exception:
            attrs[k] = str(v)
    return attrs


def preview_dataset(ds: h5py.Dataset, max_rows: int = 50, max_cols: int = 50) -> np.ndarray:
    arr = ds[()]
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        n = min(arr.shape[0], max_rows)
        return arr[:n].reshape(n, 1)
    # For 2D+, show first slice along leading axes to produce 2D
    if arr.ndim >= 2:
        # Build slice that takes first index of all axes beyond 2D
        slicer = [slice(None), slice(None)] + [0] * (arr.ndim - 2)
        view = arr[tuple(slicer)]
        r = min(view.shape[0], max_rows)
        c = min(view.shape[1], max_cols)
        return view[:r, :c]

    return arr


def main():
    st.set_page_config(page_title="HDF5 Viewer", layout="wide")
    st.title("HDF5 (.h5) Explorer")
    st.caption("Browse groups, datasets, attributes, and preview values.")

    default_path = "distilbert_model_weights.h5" if os.path.exists("distilbert_model_weights.h5") else None

    with st.sidebar:
        st.header("Choose Source")
        source = st.radio("File source", ["Existing file", "Upload", "From URL"],
                          index=0 if default_path else 1)
        uploaded_file = None
        chosen_path: Optional[str] = None
        url_input: Optional[str] = None

        if source == "Existing file":
            st.write("Pick a local .h5 present in this folder")
            candidates = [f for f in os.listdir(".") if f.lower().endswith((".h5", ".hdf5"))]
            if candidates:
                # Prefer default if available
                index = candidates.index(os.path.basename(default_path)) if default_path and os.path.basename(default_path) in candidates else 0
                chosen = st.selectbox("Select file", candidates, index=index)
                chosen_path = os.path.abspath(chosen)
            else:
                st.info("No .h5 files found next to the app.")
        elif source == "Upload":
            uploaded_file = st.file_uploader("Upload .h5 / .hdf5", type=["h5", "hdf5"]) 
        else:
            url_input = st.text_input("Direct URL to .h5/.hdf5/.keras file")
            fetch = st.button("Fetch file from URL")

    h5_handle: Optional[h5py.File] = None
    tmp_file_path: Optional[str] = None

    if uploaded_file is not None:
        # Persist to a temporary file because h5py needs a real file path for some ops
        tmp = st.session_state.get("_tmp_upload_path")
        if not tmp or not os.path.exists(tmp):
            tmp = os.path.abspath("_uploaded.h5")
            with open(tmp, "wb") as fh:
                fh.write(uploaded_file.getbuffer())
            st.session_state["_tmp_upload_path"] = tmp
        tmp_file_path = tmp

    # URL download handling
    if source == "From URL" and url_input and (st.session_state.get("_last_url") != url_input or st.session_state.get("_url_path") is None):
        if 'fetch' in locals() and fetch:
            try:
                st.info("Downloading file...")
                resp = requests.get(url_input, stream=True, timeout=60)
                resp.raise_for_status()
                total = int(resp.headers.get('content-length', 0))
                path = os.path.abspath("downloaded.h5")
                with open(path, 'wb') as f_out:
                    downloaded = 0
                    chunk = max(total // 100, 1024 * 64)
                    progress = st.progress(0)
                    for chunk_data in resp.iter_content(chunk_size=chunk):
                        if chunk_data:
                            f_out.write(chunk_data)
                            downloaded += len(chunk_data)
                            if total > 0:
                                progress.progress(min(downloaded / total, 1.0))
                st.success("Download complete.")
                st.session_state["_url_path"] = path
                st.session_state["_last_url"] = url_input
            except Exception as e:
                st.error(f"Failed to download: {e}")
    url_path = st.session_state.get("_url_path")

    file_path = tmp_file_path or url_path or chosen_path

    if not file_path:
        st.warning("Please select or upload an HDF5 file from the sidebar.")
        st.stop()

    st.subheader("File Summary")
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("File", os.path.basename(file_path))
    with colB:
        try:
            st.metric("Size (MB)", f"{os.path.getsize(file_path)/1e6:.2f}")
        except Exception:
            st.metric("Size (MB)", "-")
    with colC:
        st.metric("Path", file_path)

    # Detect file type by magic bytes
    def _first8(path: str) -> bytes:
        try:
            with open(path, 'rb') as fh:
                return fh.read(8)
        except Exception:
            return b''

    magic = _first8(file_path)
    HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
    ZIP_MAGIC = b"PK\x03\x04"

    if magic.startswith(HDF5_MAGIC):
        file_type = "hdf5"
    elif magic.startswith(ZIP_MAGIC):
        file_type = "zip"
    else:
        file_type = "unknown"

    if file_type == "zip":
        st.warning("This file is a ZIP archive, not an HDF5 file. If this is a Keras 3 model, consider renaming to .keras.")
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                st.markdown("### ZIP Contents")
                infos = zf.infolist()
                st.write(f"Entries: {len(infos)}")
                for info in infos[:500]:
                    st.write(f"- {info.filename} ({info.file_size} bytes)")
        except Exception as e:
            st.error(f"Failed to read ZIP archive: {e}")
        st.stop()

    if file_type != "hdf5":
        st.error("Unsupported file format for HDF5 viewer. Please provide a valid .h5/.hdf5 file.")
        st.stop()

    try:
        h5_handle = h5py.File(file_path, "r")
    except OSError as e:
        st.error(f"Failed to open HDF5 file: {e}")
        st.stop()

    with h5_handle as f:
        # Collect datasets and optional summary
        dataset_paths = list_datasets(f)

        left, right = st.columns([1, 2])

        with left:
            st.markdown("### Datasets")
            if not dataset_paths:
                st.info("No datasets found. This file may contain only groups/attributes.")
            selected = st.selectbox("Choose a dataset path", dataset_paths) if dataset_paths else None

            st.markdown("### Groups")
            with st.expander("Explore groups structure", expanded=False):
                def render_group(path: str, grp: h5py.Group, depth: int = 0):
                    children: List[Tuple[str, str]] = []  # (name, type)
                    for k, v in grp.items():
                        full = f"{path}/{k}" if path != "/" else f"/{k}"
                        if isinstance(v, h5py.Group):
                            children.append((full, "group"))
                        elif isinstance(v, h5py.Dataset):
                            children.append((full, "dataset"))
                    children.sort(key=lambda x: x[0])
                    for full, typ in children:
                        label = f"{full} ({typ})"
                        st.write("\u2514\ufe0f" + "\u2500" * min(1 + depth, 5) + " " + label)
                        if typ == "group":
                            render_group(full, f[full], depth + 1)

                render_group("/", f["/"])

        with right:
            st.markdown("### Details")
            if selected:
                ds = f[selected]
                st.write(f"Path: `{selected}`")
                st.write(f"Shape: {tuple(ds.shape)} | Dtype: {ds.dtype}")
                attrs = get_object_attrs(ds)
                if attrs:
                    with st.expander("Attributes", expanded=False):
                        st.json(attrs)

                # Preview
                try:
                    arr = preview_dataset(ds)
                    st.markdown("#### Preview")
                    # Make 2D table
                    arr2d = arr if arr.ndim == 2 else np.atleast_2d(arr)
                    # Column names for display
                    cols = [f"c{i}" for i in range(arr2d.shape[1])]
                    import pandas as pd
                    df = pd.DataFrame(arr2d, columns=cols)
                    st.dataframe(df, use_container_width=True)

                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        buf = io.BytesIO()
                        np.save(buf, arr)
                        st.download_button(
                            label="Download as .npy",
                            data=buf.getvalue(),
                            file_name=os.path.basename(selected).replace("/", "_") + ".npy",
                            mime="application/octet-stream",
                        )
                    with col2:
                        import pandas as pd
                        # Flatten for CSV if needed
                        flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)
                        csv = pd.DataFrame(flat).to_csv(index=False)
                        st.download_button(
                            label="Download as .csv",
                            data=csv,
                            file_name=os.path.basename(selected).replace("/", "_") + ".csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    st.warning(f"Preview unavailable: {e}")

            # Optional summary panel
            if summarize_h5:
                with st.expander("File summary (quick)", expanded=False):
                    try:
                        summ = summarize_h5(file_path, limit=50)
                        st.json(summ)
                    except Exception as e:
                        st.write(f"Summary failed: {e}")


if __name__ == "__main__":
    main()
