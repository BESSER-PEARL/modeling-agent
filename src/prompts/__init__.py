"""The text the agent sends to the model, one module per diagram type or state.

A prompt that never changes is a constant (``class_diagram.MODIFY_SYSTEM_PROMPT_CLASS``);
a prompt built from the request is a function of exactly those values
(``class_diagram.reasoning_prompt(user_request)``). Callers import the module
(``from prompts import bpmn as bpmn_prompts``) and look the prompt up by name.

Edit wording here, never at a call site. System prompts are composed at import
time and must stay byte-identical across calls, or the provider's prompt cache
misses: nothing date-, uuid- or request-dependent goes into a constant.
``tests/test_prompt_snapshots.py`` pins a hash of every prompt, so a change
shows up as a failing test; update its hash in the same commit.

Prompts may embed schema vocabulary (stereotype descriptions, reply types), so
this layer sits right above ``schemas``.
"""
