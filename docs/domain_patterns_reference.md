# Domain & State Pattern Hints (Disabled)

Domain and state pattern hints were removed from the active execution flow. The LLM (GPT-4.1) produces good diagrams without them, and the pattern injection had issues:

- Matched against conversation history instead of just the current request
- Biased the LLM toward hardcoded templates instead of letting it design freely
- Only covered known domains (library, e-commerce, hotel, etc.) — unknown domains got no help
- Added maintenance burden for every new domain

The pattern data files are still in the codebase and can be re-enabled if needed.

## Files

- `src/domain_patterns.py` — Class diagram domain patterns (library, e-commerce, hotel, hospital, etc.)
- `src/state_patterns.py` — State machine patterns (order processing, authentication, document workflow, etc.)

## How to re-enable

### In `src/execution.py`

Import the pattern functions:

```python
from domain_patterns import get_pattern_hint
from state_patterns import get_state_pattern_hint
```

Compute the hint from `operation_request` (not the full `modeling_prompt`):

```python
if target_diagram_type == "StateMachineDiagram":
    _domain_hint = get_state_pattern_hint(operation_request)
else:
    _domain_hint = get_pattern_hint(operation_request)
```

Pass it to the handler:

```python
result = handler.generate_complete_system(
    modeling_prompt,
    existing_model=target_model,
    domain_hint=_domain_hint,
)
```

### In the handler (`class_diagram_handler.py` or `state_machine_handler.py`)

```python
from domain_patterns import get_pattern_hint  # or state_patterns

# Inside generate_complete_system():
pattern_hint = kwargs.get("domain_hint", "")
if pattern_hint:
    system_prompt += "\n\n" + pattern_hint
```

**Important**: Always compute the hint from the clean `operation_request` in `execution.py`, not inside the handler where `user_request` contains the full prompt with conversation history.
