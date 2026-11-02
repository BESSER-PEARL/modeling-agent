"""Deployment diagram prompts: design rules, reasoning and modification."""

from schemas.deployment_diagram import DEPLOY_COMPONENT_STEREOTYPE_DESCRIPTION, NODE_STEREOTYPE_DESCRIPTION

from .fragments import ARCHITECTURE_MODIFY_SHARED_RULES, POSITION_DISCLAIMER


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


SYSTEM_PROMPT = f"""You are a software deployment architecture expert. Create a UML Deployment Diagram from the user's request.

DESIGN RULES:
1. Use DeploymentNode for execution environments: physical servers, virtual machines, Docker hosts, cloud services. Node stereotypes — {NODE_STEREOTYPE_DESCRIPTION}
2. Use DeploymentArtifact for the deployed software package hosted INSIDE a node (owner = the node id).
3. Use DeploymentComponent for the logical software unit an artifact implements; place it OUTSIDE nodes and link it to its artifact via manifestedBy. Component stereotypes — {DEPLOY_COMPONENT_STEREOTYPE_DESCRIPTION}
4. Use DeploymentDependency for communication paths between artifacts or components.
5. Node names are clear and descriptive ('Production Server', 'Docker Host', 'AWS Lambda'); artifact names match the software ('WebApp', 'APIGateway', 'PostgresDB').
6. Keep it focused (typically 2-5 nodes, 3-8 artifacts).
7. {POSITION_DISCLAIMER}

Element ids are short lowercase slugs (e.g. 'prod_server', 'webapp_artifact') referenced by dependencies."""


def reasoning_prompt(request: str) -> str:
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
