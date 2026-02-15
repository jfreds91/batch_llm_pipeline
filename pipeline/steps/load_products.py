"""Step 0: Load products from the catalog into the pipeline state.

NOTE: this is a special case that does not follow the BaseStep
contract because it is loading seed items from scratch - it has no input,
makes no requests, does no validation, etc. Just inits a state dict.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pipeline.catalog import CATALOG
from pipeline.models import ITEM_DATA, METADATA, Product

logger = logging.getLogger(__name__)


def load_products() -> dict[str, Any]:
    """Load all products from the catalog and seed the pipeline state.

    Returns:
        A pipeline state dict with items and metadata (including run_id).
    """
    run_id = str(uuid.uuid4())
    items: list[dict[str, Any]] = []
    for sku, entry in CATALOG.items():
        product = Product(
            sku=sku,
            name=entry["name"],
            category=entry["category"],
            material=entry["material"],
            price=entry["price"],
        )
        items.append(product.to_state_dict())
    logger.info("[load_products] run_id=%s, %d products loaded", run_id, len(items))
    return {
        ITEM_DATA: items,
        METADATA: {"run_id": run_id},
    }
