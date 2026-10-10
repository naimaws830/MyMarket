"""Stage two relevance filter: embedding similarity against product descriptions.

PRD 6.2 stage 2. The alias stage is literal and lets through articles that use
a product word in an unrelated sense, such as a turtle laying eggs. This stage
compares the article against short sentences describing what the product topic
actually is, and drops anything below the tuned threshold.

The model is loaded lazily so importing this module stays cheap, which matters
because the MCP server imports it at startup.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

from mymarket import config

log = logging.getLogger(__name__)

MAX_CHARS = 2000


@lru_cache(maxsize=1)
def _model():
    """Load the sentence transformer once per process."""
    from sentence_transformers import SentenceTransformer

    log.info("loading embedding model %s", config.EMBED_MODEL)
    return SentenceTransformer(config.EMBED_MODEL)


def available() -> bool:
    """Whether the embedding model can be loaded, without raising if it cannot."""
    try:
        _model()
        return True
    except Exception as exc:  # noqa: BLE001 - absence must degrade, not crash
        log.warning("embedding model unavailable: %s", exc)
        return False


def encode(texts: Sequence[str]):
    """Encode texts with the e5 query prefix and unit norm."""
    import numpy as np

    if not texts:
        return np.zeros((0, 384), dtype="float32")
    prefixed = [config.EMBED_PREFIX + (t or "")[:MAX_CHARS] for t in texts]
    return _model().encode(prefixed, normalize_embeddings=True, show_progress_bar=False)


@lru_cache(maxsize=1)
def _product_vectors() -> dict[str, object]:
    """Embed the description sentences of every product once."""
    cfg = config.settings()
    out: dict[str, object] = {}
    for product in cfg.products.values():
        sentences = product.description_sentences or [f"The price of {product.label} in Malaysia."]
        out[product.key] = encode(sentences)
    return out


@lru_cache(maxsize=1)
def _distractor_vectors():
    """Embed the shared distractor sentences once."""
    return encode(config.settings().distractor_sentences)


def _margin(vector, product_vectors) -> float:
    """Best product similarity minus best distractor similarity."""
    import numpy as np

    if product_vectors is None or len(product_vectors) == 0:
        return 0.0
    positive = float(np.max(np.asarray(product_vectors) @ vector))

    distractors = _distractor_vectors()
    if distractors is None or len(distractors) == 0:
        return positive
    negative = float(np.max(np.asarray(distractors) @ vector))
    return positive - negative


def score(text: str, product: str) -> float:
    """Contrastive relevance score for one article and product.

    Not a plain cosine. It is the best similarity to the product description
    sentences minus the best similarity to the distractor sentences, so a
    recipe or a wildlife story scores near zero even though it talks about the
    product word. See the filter notes in config/products.yaml.
    """
    import numpy as np

    article = encode([text])
    if len(article) == 0:
        return 0.0
    return _margin(np.asarray(article[0]), _product_vectors().get(product))


def score_many(text: str, products: Sequence[str]) -> dict[str, float]:
    """Score one article against several products with a single encode."""
    import numpy as np

    if not products:
        return {}
    article = encode([text])
    if len(article) == 0:
        return {p: 0.0 for p in products}

    vector = np.asarray(article[0])
    table = _product_vectors()
    return {p: _margin(vector, table.get(p)) for p in products}


def keeps(text: str, product: str, threshold: float | None = None) -> bool:
    """Whether the article clears the similarity threshold for the product."""
    limit = config.settings().embed_threshold if threshold is None else threshold
    return score(text, product) >= limit
