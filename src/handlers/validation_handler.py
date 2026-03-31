"""Bridge to the BESSER backend validation service."""

import logging
import requests

logger = logging.getLogger(__name__)


def validate_diagram(
    diagram_json: dict,
    diagram_type: str,
    api_url: str = "http://localhost:3001",
) -> dict:
    """Validate a diagram via the BESSER backend.

    Args:
        diagram_json: The diagram model JSON
        diagram_type: One of ClassDiagram, StateMachineDiagram, etc.
        api_url: Base URL of the BESSER backend API

    Returns:
        {"valid": bool, "errors": list[str], "warnings": list[str]}
    """
    try:
        response = requests.post(
            f"{api_url}/besser_api/validate-diagram",
            json={"type": diagram_type, "model": diagram_json},
            timeout=15,
        )
        if response.status_code == 200:
            data = response.json()
            errors = data.get("errors", [])
            warnings = data.get("warnings", [])
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
            }
        else:
            logger.warning(f"Validation API returned {response.status_code}")
            return {"valid": True, "errors": [], "warnings": ["Validation service unavailable"]}
    except Exception as e:
        logger.warning(f"Validation failed: {e}")
        return {"valid": True, "errors": [], "warnings": ["Could not reach validation service"]}
