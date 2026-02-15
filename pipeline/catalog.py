"""Mock product catalog."""

from __future__ import annotations

CATALOG: dict[str, dict[str, str | float]] = {
    "OAK-TBL-001": {
        "name": "Oakwood Dining Table",
        "category": "furniture",
        "material": "solid oak",
        "price": 1299.00,
    },
    "WLN-CHR-002": {
        "name": "Walnut Lounge Chair",
        "category": "furniture",
        "material": "walnut veneer with linen upholstery",
        "price": 849.00,
    },
    "MBL-LMP-003": {
        "name": "Marble Accent Lamp",
        "category": "lighting",
        "material": "carrara marble base with brass fittings",
        "price": 329.00,
    },
    "CER-VSE-004": {
        "name": "Ceramic Floor Vase",
        "category": "decor",
        "material": "hand-glazed stoneware",
        "price": 189.00,
    },
    "TEK-BKC-005": {
        "name": "Teak Bookcase",
        "category": "furniture",
        "material": "reclaimed teak",
        "price": 1749.00,
    },
}
