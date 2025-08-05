# app/utils.py
import pandas as pd
from scipy.sparse import hstack, csr_matrix

def preprocess_input(df, preprocessor, tfidf_vectorizers):
    # Separate text and non-text columns
    possible_text_cols = ['description', 'company_profile', 'requirements', 'benefits']
    text_cols = [col for col in possible_text_cols if col in df.columns]
    non_text_cols = [col for col in df.columns if col not in text_cols]

    # Process tabular data
    tabular = preprocessor.transform(df[non_text_cols])
    tabular_sparse = csr_matrix(tabular)

    # Process text
    text_vectors = []
    for col in text_cols:
        tfidf = tfidf_vectorizers[col]
        vec = tfidf.transform(df[col].fillna(""))
        text_vectors.append(vec)

    # Combine
    if text_vectors:
        text_combined = hstack(text_vectors)
        return hstack([tabular_sparse, text_combined])
    else:
        return tabular_sparse
