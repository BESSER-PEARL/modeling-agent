"""The deployment's model catalog, shared with the BESSER backend.

``BESSER_MODEL_CATALOG`` points at the JSON file the backend also reads (see
BESSER's ``docs/source/spec_driven_agent/models.rst``); mount the same file
into both services. The agent takes two things from it:

* the endpoint of a gateway provider (one with a ``base_url``, e.g. ``pia``):
  the frontend sends only the provider id, so the host stays server-side;
* the rates a model bills, per gateway host or (for a provider without a
  ``base_url``) on the vendor's own API. Prices are USD per 1M tokens.

Unset, the catalog is empty: no gateway, and every cost comes from the agent's
own vendor price table.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

CATALOG_ENV_VAR = "BESSER_MODEL_CATALOG"

Rates = Dict[str, float]


@dataclass(frozen=True)
class ModelCatalog:
    gateways: Dict[str, str] = field(default_factory=dict)
    # host -> lower-cased model id -> rates
    route_prices: Dict[str, Dict[str, Rates]] = field(default_factory=dict)
    # lower-cased model id -> rates on the vendor's own API
    listed_prices: Dict[str, Rates] = field(default_factory=dict)

    def base_url(self, provider: str) -> Optional[str]:
        return self.gateways.get((provider or "").strip().lower())

    def price(self, model: str, route: Optional[str]) -> Optional[Rates]:
        """The catalog rate for *model* on *route* (a base URL, or None for a
        vendor API). A gateway the catalog prices never falls back to a vendor
        rate: the same id costs differently there."""
        model_lower = (model or "").strip().lower()
        host = (urlparse(route).hostname or route).lower() if route else ""
        if host in self.route_prices:
            return self.route_prices[host].get(model_lower)
        return self.listed_prices.get(model_lower)


def _rates(raw: dict, where: str) -> Rates:
    try:
        rates = {key: float(raw[key]) for key in ("input", "output")}
        for key in ("cache_read", "cache_write"):
            rates[key] = float(raw[key]) if raw.get(key) is not None else rates["input"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Model catalog {where}: price needs numeric input and output") from exc
    return rates


def _build(data: dict) -> ModelCatalog:
    catalog = ModelCatalog()
    for provider_id, entry in (data.get("providers") or {}).items():
        if not isinstance(entry, dict):
            continue  # null removes a provider in the backend's merge
        base_url = entry.get("base_url")
        priced = {
            str(model["id"]).strip().lower(): _rates(model["price"], f"providers.{provider_id}")
            for model in entry.get("models") or []
            if isinstance(model, dict) and model.get("id") and model.get("price") is not None
        }
        if base_url:
            host = urlparse(base_url).hostname
            if not host:
                raise ValueError(f"Model catalog providers.{provider_id}: bad base_url")
            catalog.gateways[provider_id.lower()] = base_url
            catalog.route_prices.setdefault(host.lower(), {}).update(priced)
        else:
            catalog.listed_prices.update(priced)
    for host, table in (data.get("route_prices") or {}).items():
        if host.startswith("_") or not isinstance(table, dict):
            continue
        catalog.route_prices.setdefault(host.lower(), {}).update({
            model.lower(): _rates(rate, f"route_prices.{host}")
            for model, rate in table.items() if not model.startswith("_")
        })
    return catalog


@lru_cache(maxsize=1)
def load_model_catalog() -> ModelCatalog:
    """The catalog named by ``BESSER_MODEL_CATALOG``, read once per process.

    Raises ``ValueError`` for an unreadable or malformed file, so a broken
    deployment catalog stops the agent at startup.
    """
    path = os.getenv(CATALOG_ENV_VAR, "").strip()
    if not path:
        return ModelCatalog()
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Model catalog {path} cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Model catalog {path} must be a JSON object")
    catalog = _build(data)
    logger.info("Model catalog %s: gateways %s", path, sorted(catalog.gateways))
    return catalog
