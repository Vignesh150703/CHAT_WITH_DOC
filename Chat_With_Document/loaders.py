import os
import tempfile
from typing import List

import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_unstructured import UnstructuredLoader


def load_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]):
    all_docs = []
    failed_files = []

    for uf in uploaded_files:
        suffix = "." + uf.name.split(".")[-1] if "." in uf.name else ""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.getbuffer())
                tmp_path = tmp.name

            ext = suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif ext == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif ext == ".docx":
                loader = Docx2txtLoader(tmp_path)
            else:
                loader = UnstructuredLoader(tmp_path)

            docs = loader.load()
            if docs:
                for i, d in enumerate(docs):
                    d.metadata.setdefault("source_file", uf.name)
                    if "page" not in d.metadata:
                        d.metadata["page"] = i + 1
                all_docs.extend(docs)
            else:
                failed_files.append(f"{uf.name} (no content extracted)")
        except Exception as e:
            failed_files.append(f"{uf.name}: {str(e)}")
            continue
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    if failed_files:
        st.warning(f"⚠️ Some files could not be loaded: {', '.join(failed_files)}")

    return all_docs

