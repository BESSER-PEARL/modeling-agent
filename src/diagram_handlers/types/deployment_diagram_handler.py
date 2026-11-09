"""
Deployment Diagram Handler
Handles generation and modification of UML Deployment Diagrams.

Elements: DeploymentNode (execution environment), DeploymentArtifact (physical artifact),
DeploymentComponent (logical component).
Relationships: DeploymentDependency.
Positions are NOT generated here — the WME converter places the elements.
"""

import logging
from typing import Any, Dict

from ..core.architecture_handler import ArchitectureDiagramHandler
from prompts import deployment_diagram as deployment_prompts
from schemas import DeploymentModificationResponse, SystemDeploymentSpec
from protocol.actions import inject_complete_system

logger = logging.getLogger(__name__)


class DeploymentDiagramHandler(ArchitectureDiagramHandler):
    """Handler for UML Deployment Diagram generation and modification."""

    _LABEL = "deployment diagram"
    _SYSTEM_FAILURE_MESSAGE = "I couldn't generate that deployment diagram. Please try again or rephrase your request."
    _SYSTEM_SCHEMA = SystemDeploymentSpec
    _MODIFICATION_SCHEMA = DeploymentModificationResponse
    _MODIFY_SYSTEM_PROMPT = deployment_prompts.MODIFY_SYSTEM_PROMPT_DEPLOYMENT
    _MODIFY_EXAMPLES = "*'add a Docker Host node'* or *'rename Production Server to AWS EC2'*"
    _REF_ADD_ACTIONS = {
        "add_node": "DeploymentNode",
        "add_artifact": "DeploymentArtifact",
        "add_component": "DeploymentComponent",
    }
    _REF_NAMED_TYPES = frozenset({"DeploymentNode", "DeploymentArtifact", "DeploymentComponent"})

    def get_diagram_type(self) -> str:
        return "DeploymentDiagram"

    def get_system_prompt(self) -> str:
        return deployment_prompts.SYSTEM_PROMPT

    def _reasoning_prompt(self, request: str) -> str:
        return deployment_prompts.reasoning_prompt(request)

    # ------------------------------------------------------------------
    # Single element + fallbacks
    # ------------------------------------------------------------------

    _STARTER_DEFAULT_NAME = "Server"

    def _starter_element(self, name: str) -> Dict[str, Any]:
        return inject_complete_system(
            {
                "systemName": name,
                "nodes": [{"id": "node1", "name": name, "stereotype": "node"}],
                "artifacts": [],
                "deployComponents": [],
                "dependencies": [],
            },
            self.get_diagram_type(),
            f"Created a starter **{name}** node. Describe the full deployment topology and I'll build it out!",
        )

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
        return inject_complete_system(
            fallback,
            self.get_diagram_type(),
            (
                "I created a starter deployment diagram. Describe your infrastructure "
                "(e.g. *'a three-tier deployment with load balancer, app servers, and database cluster'*) "
                "and I'll build a richer model!"
            ),
        )

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
