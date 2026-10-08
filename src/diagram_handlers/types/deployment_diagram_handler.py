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

from ..core.architecture_handler import ARCHITECTURE_MODIFY_SHARED_RULES, ArchitectureDiagramHandler
from ..core.prompt_fragments import POSITION_DISCLAIMER
from schemas import DeploymentModificationResponse, SystemDeploymentSpec
from schemas.deployment_diagram import DEPLOY_COMPONENT_STEREOTYPE_DESCRIPTION, NODE_STEREOTYPE_DESCRIPTION

logger = logging.getLogger(__name__)


MODIFY_SYSTEM_PROMPT_DEPLOYMENT = f"""You are a deployment diagram modeling expert. The user wants to modify a UML Deployment Diagram.

READING THE CONTEXT:
Each element appears as: [id] Name (Node:stereotype), [id] Name (Artifact [in NodeName]) or [id] Name (Component:stereotype)
Each dependency appears as: Dependency: [src-id] Name ---> [tgt-id] Name

MODIFICATION RULES:
1. Actions: "add_node", "add_artifact", "add_component", "add_dependency", "modify_element", "remove_element", "remove_dependency"
2. add_node: set target.elementName to the new node's name. Optional changes.stereotype — {NODE_STEREOTYPE_DESCRIPTION}
3. add_artifact: set target.elementName to the new artifact's name and changes.owner to the hosting node's name or [id].
4. add_component: set target.elementName to the new logical component's name. Optional changes.stereotype — {DEPLOY_COMPONENT_STEREOTYPE_DESCRIPTION}
5. add_dependency: optional changes.label (e.g. 'HTTPS').
6. modify_element: put the new name in changes.name and/or the new stereotype in changes.stereotype.

{ARCHITECTURE_MODIFY_SHARED_RULES}"""


class DeploymentDiagramHandler(ArchitectureDiagramHandler):
    """Handler for UML Deployment Diagram generation and modification."""

    _LABEL = "deployment diagram"
    _SYSTEM_SCHEMA = SystemDeploymentSpec
    _MODIFICATION_SCHEMA = DeploymentModificationResponse
    _MODIFY_SYSTEM_PROMPT = MODIFY_SYSTEM_PROMPT_DEPLOYMENT
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
        return f"""You are a software deployment architecture expert. Create a UML Deployment Diagram from the user's request.

DESIGN RULES:
1. Use DeploymentNode for execution environments: physical servers, virtual machines, Docker hosts, cloud services. Node stereotypes — {NODE_STEREOTYPE_DESCRIPTION}
2. Use DeploymentArtifact for the deployed software package hosted INSIDE a node (owner = the node id).
3. Use DeploymentComponent for the logical software unit an artifact implements; place it OUTSIDE nodes and link it to its artifact via manifestedBy. Component stereotypes — {DEPLOY_COMPONENT_STEREOTYPE_DESCRIPTION}
4. Use DeploymentDependency for communication paths between artifacts or components.
5. Node names are clear and descriptive ('Production Server', 'Docker Host', 'AWS Lambda'); artifact names match the software ('WebApp', 'APIGateway', 'PostgresDB').
6. Keep it focused (typically 2-5 nodes, 3-8 artifacts).
7. {POSITION_DISCLAIMER}

Element ids are short lowercase slugs (e.g. 'prod_server', 'webapp_artifact') referenced by dependencies."""

    def _reasoning_prompt(self, request: str) -> str:
        return (
            "You are a deployment architecture expert. Think step by step about the "
            "following deployment request and plan it before producing JSON.\n\n"
            f"User Request: {request}\n\n"
            "Analyze:\n"
            "1. What execution environments (nodes) are needed?\n"
            "2. What artifacts are deployed in each node?\n"
            "3. What logical components do the artifacts implement?\n"
            "4. What communication paths (dependencies) exist between artifacts/components?\n"
            "5. What node stereotypes best describe each environment (node/device/cloud/server)?\n\n"
            "Focus on the manifestedBy links connecting artifacts to their logical components."
        )

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
