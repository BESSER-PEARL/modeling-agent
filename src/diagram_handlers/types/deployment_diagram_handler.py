"""
Deployment Diagram Handler
Handles generation and modification of UML Deployment Diagrams.

Elements: DeploymentNode (execution environment), DeploymentArtifact (physical artifact),
DeploymentComponent (logical component).
Relationships: DeploymentDependency.
Positions are NOT generated here — the WME layout engine handles placement.
"""

import logging
from typing import Any, Dict, List, Optional

from ..core.base_handler import BaseDiagramHandler, LLMPredictionError
from model_config import MODEL_GENERATION_LARGE, MODEL_REASONING
from ..core.prompt_fragments import EXACT_NAMES_RULE, POSITION_DISCLAIMER, REMOVE_ELEMENT_RULE
from schemas import SystemDeploymentSpec, DeploymentModificationResponse
from utilities.model_context import detailed_model_summary

logger = logging.getLogger(__name__)


MODIFY_SYSTEM_PROMPT_DEPLOYMENT = """You are a deployment diagram modeling expert. The user wants to modify a UML Deployment Diagram.

READING THE CONTEXT:
Each element appears as: [id] Name (type/stereotype)
Each dependency appears as: Dependency: [src-id] Name ---> [tgt-id] Name

MODIFICATION RULES:
1. Actions: "add_node", "add_artifact", "add_component", "add_dependency", "modify_element", "remove_element", "remove_dependency"
2. add_node: set target.elementName to the node name. Optional changes.stereotype ('node','device','cloud','server').
3. add_artifact: set target.elementName to the artifact name. Set changes.owner to the node name/id that hosts it.
4. add_component: set target.elementName to the component name. Optional changes.stereotype.
5. add_dependency: set changes.source and changes.target to element name/id. Optional changes.label.
6. modify_element: set target.elementId or target.elementName. Put new name in changes.name.
7. remove_element: set target.elementId or target.elementName.
8. remove_dependency: set changes.source and changes.target.

When element not found, set elementFound: false, modifications: [], explain in message.
If user says 'undo': modifications: [], elementFound: false, message: 'To undo, use Ctrl+Z or the undo button.'"""


class DeploymentDiagramHandler(BaseDiagramHandler):
    """Handler for UML Deployment Diagram generation and modification."""

    def get_diagram_type(self) -> str:
        return "DeploymentDiagram"

    def get_system_prompt(self) -> str:
        return f"""You are a software deployment architecture expert. Create a UML Deployment Diagram from the user's request.

DESIGN RULES:
1. Use DeploymentNode for execution environments: physical servers, virtual machines, Docker containers, cloud services.
2. Use DeploymentArtifact for physical deployments hosted INSIDE a node. Artifacts represent the deployed software package.
3. Use DeploymentComponent (logical) to represent the logical software unit that the artifact implements. Place these OUTSIDE nodes.
4. Link artifact to its logical component via the manifestedBy field.
5. Use DeploymentDependency to show communication paths between artifacts or components.
6. Node names are clear and descriptive ('Production Server', 'Docker Host', 'AWS Lambda').
7. Artifact names match the software ('WebApp', 'APIGateway', 'PostgresDB').
8. Keep focused (typically 2-5 nodes, 3-8 artifacts). Do NOT add positions.

Element ids are short lowercase slugs (e.g. 'prod_server', 'webapp_artifact') referenced by dependencies."""

    # ------------------------------------------------------------------
    # Complete system (the primary generation path)
    # ------------------------------------------------------------------

    def generate_complete_system(
        self,
        user_request: str,
        existing_model: Dict[str, Any] = None,
        raw_request: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a complete diagram with two-pass structured output.

        ``raw_request`` is the user message before context enrichment; it
        drives the two-pass length check and keeps the reasoning prompt lean.
        """
        system_prompt = self.get_system_prompt()
        logger.info(f"[DeploymentDiagram] generate_complete_system called with: {user_request!r}")

        reasoning_prompt = (
            "You are a deployment architecture expert. Think step by step about the "
            "following deployment request and plan it before producing JSON.\n\n"
            f"User Request: {raw_request or user_request}\n\n"
            "Analyze:\n"
            "1. What execution environments (nodes) are needed?\n"
            "2. What artifacts are deployed in each node?\n"
            "3. What logical components do the artifacts implement?\n"
            "4. What communication paths (dependencies) exist between artifacts/components?\n"
            "5. What node stereotypes best describe each environment (node/device/cloud/server)?\n\n"
            "Focus on the manifestedBy links connecting artifacts to their logical components."
        )

        try:
            parsed = self.predict_two_pass_structured(
                user_request=user_request,
                system_prompt=system_prompt,
                reasoning_prompt=reasoning_prompt,
                raw_request=raw_request,
                model=MODEL_GENERATION_LARGE,
                reasoning_model=MODEL_REASONING,
                response_schema=SystemDeploymentSpec,
            )
            system_spec = parsed.model_dump()

            return {
                "action": "inject_complete_system",
                "systemSpec": system_spec,
                "diagramType": self.get_diagram_type(),
                "message": self._build_system_message(system_spec),
            }

        except LLMPredictionError as exc:
            logger.error(f"[DeploymentDiagram] generate_complete_system LLM FAILED: {exc}")
            return self._error_response(
                "I couldn't generate that deployment diagram. Please try again or rephrase your request.",
                code="llm_failure",
            )
        except Exception as exc:
            logger.error(f"[DeploymentDiagram] generate_complete_system FAILED: {exc}", exc_info=True)
            return self.generate_fallback_system()

    # ------------------------------------------------------------------
    # Modification
    # ------------------------------------------------------------------

    def generate_modification(
        self, user_request: str, current_model: Dict[str, Any] = None, **kwargs,
    ) -> Dict[str, Any]:
        system_prompt = MODIFY_SYSTEM_PROMPT_DEPLOYMENT

        # Store elements on the instance for ref validation
        self._elements: Dict[str, Any] = {}
        if current_model and isinstance(current_model, dict):
            raw = current_model.get("elements")
            if isinstance(raw, dict):
                self._elements = raw

        context_block = ""
        if current_model and isinstance(current_model, dict):
            summary = detailed_model_summary(current_model, self.get_diagram_type())
            if summary:
                context_block = f"\n\n{summary}"

        user_prompt = f"Modify the {self.get_diagram_type()} diagram: {user_request}{context_block}"
        logger.info(f"[DeploymentDiagram] generate_modification called with: {user_request!r}")

        try:
            result = self._execute_modification(
                user_prompt, system_prompt, DeploymentModificationResponse,
            )
            return self._validate_mod_refs(result)
        except LLMPredictionError as exc:
            logger.error(f"[DeploymentDiagram] generate_modification LLM FAILED: {exc}")
            return self._error_response(
                "I couldn't process that modification. Please try again or rephrase your request.",
            )
        except Exception as exc:
            logger.error(f"[DeploymentDiagram] generate_modification FAILED: {exc}", exc_info=True)
            return {
                "action": "assistant_message",
                "message": (
                    "I couldn't apply that modification automatically. Could you rephrase it? "
                    "For example: *'add a Docker container node'* or "
                    "*'rename Production Server to AWS EC2'*."
                ),
            }

    # ------------------------------------------------------------------
    # Single element + fallbacks
    # ------------------------------------------------------------------

    def generate_single_element(
        self, user_request: str, existing_model: Dict[str, Any] = None, **kwargs,
    ) -> Dict[str, Any]:
        name = self.extract_name_from_request(user_request, "Server")
        return {
            "action": "inject_complete_system",
            "systemSpec": {
                "systemName": name,
                "nodes": [{"id": "node1", "name": name, "stereotype": "node"}],
                "artifacts": [],
                "deployComponents": [],
                "dependencies": [],
            },
            "diagramType": self.get_diagram_type(),
            "message": f"Created a starter **{name}** node. Describe the full deployment topology and I'll build it out!",
        }

    def generate_fallback_element(self, request: str) -> Dict[str, Any]:
        return self.generate_single_element(request)

    def generate_fallback_system(self) -> Dict[str, Any]:
        fallback = {
            "systemName": "BasicDeployment",
            "nodes": [
                {"id": "web_server", "name": "Web Server", "stereotype": "server"},
                {"id": "db_server", "name": "Database Server", "stereotype": "server"},
            ],
            "artifacts": [
                {"id": "webapp_artifact", "name": "WebApp", "owner": "web_server"},
                {"id": "db_artifact", "name": "PostgresDB", "owner": "db_server"},
            ],
            "deployComponents": [
                {"id": "web_comp", "name": "WebApp", "stereotype": "solution", "manifestedBy": "webapp_artifact"},
                {"id": "db_comp", "name": "PostgresDB", "stereotype": "db", "manifestedBy": "db_artifact"},
            ],
            "dependencies": [
                {"source": "webapp_artifact", "target": "db_artifact", "name": "JDBC"},
            ],
        }
        return {
            "action": "inject_complete_system",
            "systemSpec": fallback,
            "diagramType": self.get_diagram_type(),
            "message": (
                "I created a starter deployment diagram. Describe your infrastructure "
                "(e.g. *'a three-tier deployment with load balancer, app servers, and database cluster'*) "
                "and I'll build a richer model!"
            ),
        }

    # ------------------------------------------------------------------
    # Message builder
    # ------------------------------------------------------------------

    def _build_system_message(self, spec: Dict[str, Any]) -> str:
        name = spec.get("systemName") or "deployment"
        nodes = spec.get("nodes", [])
        artifacts = spec.get("artifacts", [])
        node_names = [n.get("name", "?") for n in nodes][:4]
        msg = f"Built the **{name}** deployment diagram with {len(nodes)} node(s) and {len(artifacts)} artifact(s)"
        if node_names:
            msg += f": {', '.join(f'**{n}**' for n in node_names)}"
        msg += ". Ask me to add nodes, artifacts, or communication paths!"
        return msg

    # ------------------------------------------------------------------
    # Element resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_element(ref: Optional[str], elements: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Look up a Deployment element by id (exact key) then by name (case-insensitive)."""
        if not ref or not isinstance(elements, dict):
            return None
        el = elements.get(ref)
        if isinstance(el, dict):
            return el
        lower = ref.lower()
        for el in elements.values():
            if isinstance(el, dict) and el.get("type") in ("DeploymentNode", "DeploymentArtifact", "DeploymentComponent"):
                if (el.get("name") or "").lower() == lower:
                    return el
        return None

    # ------------------------------------------------------------------
    # Server-side ref guardrail
    # ------------------------------------------------------------------

    def _ref_exists(self, mod: Dict[str, Any], elements: Dict[str, Any]) -> bool:
        """Return True if every element ref in this modification exists in the model."""
        action = mod.get("action", "")
        if action in ("modify_element", "remove_element"):
            target = mod.get("target") or {}
            ref = target.get("elementId") or target.get("elementName")
            return ref is None or self._resolve_element(ref, elements) is not None
        if action in ("add_dependency", "remove_dependency"):
            changes = mod.get("changes") or {}
            src = changes.get("source")
            tgt = changes.get("target")
            src_ok = src is None or self._resolve_element(src, elements) is not None
            tgt_ok = tgt is None or self._resolve_element(tgt, elements) is not None
            return src_ok and tgt_ok
        return True

    def _validate_mod_refs(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Drop modifications whose element refs cannot be resolved in the current model."""
        elements = self._elements
        if not elements or result.get("action") != "modify_model":
            return result

        if "modifications" in result:
            mods = result["modifications"]
            valid = [m for m in mods if self._ref_exists(m, elements)]
            dropped = len(mods) - len(valid)
            if dropped:
                logger.info(f"[DeploymentDiagram] Dropped {dropped} modification(s) with unresolved element ref(s)")
            if not valid:
                return {
                    "action": "assistant_message",
                    "message": (
                        "I couldn't find the element(s) you described in the current diagram. "
                        "Please check the names and try again."
                    ),
                }
            result = dict(result)
            result["modifications"] = valid
            return result

        if "modification" in result:
            if not self._ref_exists(result["modification"], elements):
                logger.info("[DeploymentDiagram] Dropped modification with unresolved element ref")
                return {
                    "action": "assistant_message",
                    "message": (
                        "I couldn't find that element in the current diagram. "
                        "Please check the name and try again."
                    ),
                }

        return result
