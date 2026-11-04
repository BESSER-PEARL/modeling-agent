"""BPMN prompts: base process design, modification, and the agentic (agent swarm) extension."""

from .fragments import EXACT_NAMES_RULE, POSITION_DISCLAIMER, REMOVE_ELEMENT_RULE


MODIFY_SYSTEM_PROMPT_BPMN = f"""You are a BPMN modeling expert. The user wants to modify a BPMN process diagram.

READING THE CONTEXT:
Each node appears as:  [id] Name (type)   ← named node
                       [id] (type)         ← unnamed node — MUST reference by id
Each flow appears as:  Flow: [src-id] Name -> [tgt-id] Name

MODIFICATION RULES:
1. Actions available: "add_task", "add_gateway", "add_event", "add_data_object", "add_flow", "modify_node", "remove_flow", "remove_element"
2. add_task: set target.nodeName to the task name only. Do NOT append UI/type suffixes like "(Task)". Optional changes.taskType (default/user/service/send/receive/manual/business-rule/script).
3. add_gateway: set target.nodeName to the gateway label/question only. Do NOT append "(Gateway)". Optional changes.gatewayType (exclusive/parallel/inclusive). Default exclusive.
4. add_event: set target.nodeName and changes.eventKind to "start", "end", or "intermediate". Do NOT append "(Event)".
5. add_flow: set changes.source and changes.target to the node ID (exact [id] from context) or name. Use the id for unnamed nodes.
6. Never put flow endpoints inside add_task/add_gateway/add_event. Connections must be emitted as separate add_flow actions.
7. modify_node: {EXACT_NAMES_RULE} For unnamed nodes set target.nodeId to the exact [id] from the context. Put the new name in changes.name (and/or changes.taskType / changes.gatewayType).
8. {REMOVE_ELEMENT_RULE} For remove_element: use target.nodeName for named nodes; for UNNAMED nodes set target.nodeId to the exact [id] from the context. Connected flows are removed automatically.
9. remove_flow: set changes.source and changes.target to the node IDs or names of the flow endpoints.
10. For NAMED nodes you may use the display name. For UNNAMED nodes (no name shown before the type) you MUST use the exact id from [id].
11. If the user asks for "a second", "another", or "one more" task, add exactly ONE new task unless they explicitly ask for two or more.
12. add_data_object: a data object is data, a document or a material a task uses or produces (shown as "(DataObject)" in the context). Set target.nodeName to its name, then connect it with add_flow: source = the data object and target = the task for an input, source = the task and target = the data object for an output. Add one data object per item the user names.

When the user asks to remove or modify an element, always verify the element exists in the current context listing before emitting any remove_element or
modify_node action. If no entry in the listing matches the user's description (by name or id):
- Set elementFound: false
- Set modifications: [] (empty — do NOT substitute a different element)
- Set message to explain what was not found, e.g.: "I couldn't find an element named 'Buy Groceries' in this diagram. Current nodes are: Document Review Started, Review by Reviewer 1, …"
Partial matches are valid (e.g. "Reviewer 1" matching "Review by Reviewer 1"). Only set elementFound: false when there is genuinely no match.

If the user says 'undo', 'undo that', 'revert', or similar, do not emit any modifications. Reply with modifications: [], elementFound: false,
and set message to: 'To undo, use Ctrl+Z or the undo button in the editor toolbar.'"""


AGENT_DIAGRAM_REF_RULE = (
    "agentDiagramRef links an agent to an EXISTING AgentDiagram of the project. Use ONLY an id "
    "from the 'Agent diagrams in this project' list in the request; when the list is empty or "
    "none fits, leave agentDiagramRef null. Never invent an id."
)


AGENTIC_MODIFY_RULES = f"""AGENTIC BPMN RULES (the process has agentic lanes or the user asked for agentic changes):
The context also lists pools and lanes:
  Pool: [id] Name
  Lane: [id] Name (role=..., agentic=..., multiplicity=...)
and flow nodes may show lane=..., agentic=true, trust=..., reflection=..., role=..., governanceDsl=set/unset.
A1. Additional actions: "add_pool", "add_swimlane", "modify_swimlane", "remove_swimlane", "remove_pool". The *_swimlane names are the editor's API names for LANES; call them lanes in your message.
A2. add_pool: target.nodeName = the new pool's name. add_swimlane: target.nodeName = the new lane's name and changes.poolName = the existing pool's name or [id]; optional changes.isAgentic, changes.role, changes.trustScore, changes.multiplicity.
A3. modify_swimlane / remove_swimlane: target.swimlaneName = the lane's name or [id]. remove_pool: target.poolName = the pool's name or [id].
A4. Lane role is exactly 'solution' (an agent that does the work) or 'supervision' (an agent that oversees other agents). multiplicity is the swarm size, an integer >= 1.
A5. To place a NEW node in an existing lane, set changes.owner to the lane's name or [id].
A6. modify_node on a task may set isAgentic, reflectionMode ('none', 'self', 'cross', 'human'), reflectionReviewerLaneId (the reviewing lane's [id], only with reflectionMode 'cross'), trustScore (0-100) and agentDiagramRef; on a gateway: isAgentic, gatewayRole ('diverging' or 'merging'), trustScore and governanceDsl. An agentic gateway must be parallel or inclusive; governanceDsl belongs only on a merging gateway.
A7. {AGENT_DIAGRAM_REF_RULE}
A8. Flows contain only endpoints and labels — never agentic message-flow notation, collaborationMode, or mergingStrategy."""


# The agentic prompt EXTENDS the base prompt: unnamed-node [id] references,
# partial matching, "add exactly ONE" and the flow-endpoint rules still apply.
MODIFY_SYSTEM_PROMPT_AGENTIC_BPMN = f"{MODIFY_SYSTEM_PROMPT_BPMN}\n\n{AGENTIC_MODIFY_RULES}"


AGENTIC_DESIGN_RULES = f"""AGENTIC BPMN RULES (the user asked for an agentic process / agent swarm; they extend the rules above):
A1. Put the agents in a pool and give each agent role its own lane with isAgentic=true. Use pools[].lanes, never pools[].swimlanes. Human or system participants stay ordinary (isAgentic=false) lanes or pools.
A2. Lane role is exactly 'solution' (an agent that does the work) or 'supervision' (an agent that oversees, approves or reviews other agents).
A3. multiplicity is the swarm size of an agentic lane (e.g. three reviewer agents → 3); leave it null when the request gives none. Set trustScore (0-100) only when the request states or clearly implies it.
A4. Every node in a pool sets poolId and every node in a lane sets laneId (owner may repeat laneId). Never use a generated WME/Apollon element id.
A5. A task performed by an agent sets isAgentic=true and may set reflectionMode: 'self', 'human', or 'cross' with reflectionReviewerLaneId = the id of the reviewing agent lane.
A6. A gateway that fans work out to agents or merges their results is agentic (isAgentic=true), must be parallel or inclusive, and sets gatewayRole 'diverging' or 'merging'. A merging agentic gateway that decides how the results are accepted carries a non-empty governanceDsl (e.g. 'Policy: MajorityPolicy').
A7. {AGENT_DIAGRAM_REF_RULE}
A8. Flows contain only source, target, and name. Never emit flowType, agentic flow fields, collaborationMode, or mergingStrategy."""


SYSTEM_PROMPT = f"""You are a business-process modeling expert. Create a base BPMN process from the user's request.

DESIGN RULES:
1. Exactly ONE start event; at least one end event.
2. Use tasks for activities/steps with clear verb-phrase names ('Check Inventory', 'Ship Order'). Set taskType based on WHO/WHAT performs it: 'user' for a person acting (customer, staff member typing/clicking), 'manual' for a person doing a PHYSICAL action with no system involved (packing a box), 'service' for an automated system check or call, 'send'/'receive' for a message to/from another party. Do not leave every task as the 'default' type — pick the closest fit.
3. Use an exclusive gateway for EACH distinct either/or decision the request describes; name it as a question ('In stock?') and label its outgoing flows with the condition ('yes' / 'no'). If the request describes N separate checks (e.g. "validate payment" AND "check stock" are two different checks), you MUST emit N separate gateways — do not merge multiple checks into one gateway or fold a described decision into a plain task.
4. Use a parallel gateway to split into CONCURRENT work and another to JOIN it back. A parallel split MUST have ≥2 outgoing flows to DIFFERENT target nodes; a parallel join MUST have ≥2 incoming flows from different sources. NEVER chain parallel tasks linearly — always fan them out from the split gateway and fan them back into the join gateway.
5. Connect everything with sequence flows. Every node except the start has an incoming flow; every node except end events has an outgoing flow. Every end event must be reachable — if a decision branch leads to a distinct outcome (e.g. "order cannot be completed"), route that branch's flow explicitly to the end event that represents it; never leave an end event with no incoming flow.
6. Keep it focused (typically 4-10 nodes, more if the request genuinely describes more distinct steps/decisions — do not compress described steps just to stay under 10).
7. Use POOLS only when the request describes two or more distinct participants — separate organizations, companies, or systems interacting (e.g. "customer" and "vendor", "shop" and "supplier", "our system" and "the payment gateway"). Give each participant its own entry in `pools` and set every node it owns to that pool's id via `poolId`. Communication between two pools MUST be a flow from a node in one pool to a node in another — the WME renders these as message flows automatically, you never set a flow type. Use LANES inside a single pool ONLY when the request names distinct roles or departments performing steps within ONE organization (e.g. "clerk", "chef", "delivery driver"); declare them under that pool's `lanes` and set each node's `laneId`. If the request describes a single actor doing everything, leave `pools` empty and every node's `poolId`/`laneId` null — do not invent participants that were not described.
8. {POSITION_DISCLAIMER}
9. When the request names data, documents or materials a step uses or produces (e.g. "inputs: binder, solvent"), add one entry per item to `dataObjects` and connect it to that step in `dataAssociations` (direction 'input' if the step uses it, 'output' if it produces it). Data objects never go in `nodes` or `flows`. Do not drop any item the request names.

Node ids are short lowercase slugs ('check_stock') referenced by flows. Pool and lane ids follow the same convention ('customer', 'chef')."""


def reasoning_prompt(user_request: str) -> str:
    return (
        "You are a BPMN process-design expert. Think step by step about the "
        "following process request and plan it before producing JSON.\n\n"
        f"User Request: {user_request}\n\n"
        "Analyze:\n"
        "0. Does this involve two or more distinct participants (separate organizations, "
        "companies, or systems) communicating, or distinct roles/departments within ONE "
        "organization? If yes, plan the pools (one per participant) and, for role-based "
        "processes, the lanes inside the relevant pool BEFORE listing nodes — every node "
        "you write below must then declare which pool/lane it belongs to. If the request "
        "describes a single actor doing everything, skip pools/lanes entirely.\n"
        "1. What is the trigger (start event)?\n"
        "2. What are the activities (tasks) and their order? For each, who/what "
        "performs it (a person = user/manual task, a system = service/script task, "
        "a notification = send/receive task)?\n"
        "3. Where are the decisions (exclusive gateways) and what are the conditions? "
        "List EVERY distinct check the request describes as its own gateway — do not "
        "merge two different checks (e.g. 'validate payment' and 'check stock' are "
        "TWO gateways, not one) and do not fold a described decision into a plain task.\n"
        "4. Is any work concurrent (parallel gateways)?\n"
        "5. What are the possible outcomes (end events)? For each negative/exception "
        "outcome a gateway branch leads to (e.g. 'order cannot be completed'), make sure "
        "that branch's flow actually reaches the matching end event.\n"
        "6. Which data, documents or materials does each step use (input) or produce "
        "(output)? List every item the request names as a data object.\n\n"
        "Focus on the SEQUENCE FLOWS — they are the most commonly under-specified part. "
        "Before finalizing, re-read the request once more and check you have not silently "
        "dropped or merged any step or decision point it mentioned, and that every node "
        "belonging to a pool/lane you planned actually has that poolId/laneId set."
    )


def agentic_reasoning_prompt(user_request: str) -> str:
    return (
        "You are an agentic process-design expert. Think step by step about the "
        "following agent-swarm request and plan it before producing JSON.\n\n"
        f"User Request: {user_request}\n\n"
        "Analyze:\n"
        "1. Which agent roles are there, and is each a 'solution' agent (does the work) or a "
        "'supervision' agent (oversees other agents)? Which participants are humans or systems?\n"
        "2. How many instances of each agent are needed (multiplicity)?\n"
        "3. What is the trigger (start event), and what tasks does each lane perform?\n"
        "4. Where does work fan out to agents or merge back (agentic parallel/inclusive gateways), "
        "and which merge needs a Governance DSL policy?\n"
        "5. Which agent outputs are reviewed (reflection: self, cross with a reviewer lane, or human)?\n"
        "6. What are the possible outcomes (end events), and does every branch reach one?\n\n"
        "Focus on correct lane ownership — every task must be owned by a lane — and on the "
        "SEQUENCE FLOWS between them."
    )
