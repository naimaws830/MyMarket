"""Feature construction shared by training and inference.

Keeping this in one place is what stops the training features and the serving
features from drifting apart.
"""

from __future__ import annotations

import pandas as pd

from mymarket import config
from mymarket.filter.aliases import build_text, find_terms

TEXT_COLUMN = "text"


def model_text(title: str, summary: str = "", body: str = "") -> str:
    """The text the model sees, per PRD 6.3."""
    return build_text(title, summary, body)


def event_frame(texts: pd.Series) -> pd.DataFrame:
    """Binary event word columns for a column of texts."""
    words = config.settings().event_words
    columns = {
        f"ev_{word.lower().replace(' ', '_')}": texts.apply(
            lambda t, w=word: int(bool(find_terms(t or "", [w])))
        )
        for word in words
    }
    return pd.DataFrame(columns, index=texts.index)


def event_columns() -> list[str]:
    """Names of the event word columns, in a stable order."""
    return [f"ev_{w.lower().replace(' ', '_')}" for w in config.settings().event_words]


def build_matrix(frame: pd.DataFrame, embeddings, ohe, fit: bool = False):
    """Stack embeddings, the product one hot and the event flags into one matrix.

    `ohe` is a fitted OneHotEncoder when `fit` is False, which is the serving
    path. The column order is embeddings, then product, then event flags, and
    it must stay that way for a saved model to keep working.
    """
    import numpy as np
    import scipy.sparse as sp

    products = frame[["product"]]
    product_block = ohe.fit_transform(products) if fit else ohe.transform(products)

    columns = event_columns()
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        events = event_frame(frame[TEXT_COLUMN].fillna(""))
        frame = frame.join(events[[c for c in missing if c in events.columns]])
    event_block = frame.reindex(columns=columns).fillna(0).to_numpy(dtype="float32")

    return sp.hstack(
        [sp.csr_matrix(np.asarray(embeddings)), sp.csr_matrix(product_block), sp.csr_matrix(event_block)]
    ).tocsr()
