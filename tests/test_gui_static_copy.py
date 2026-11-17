"""Static copy of a designed screen must not pose as data.

With a class diagram the screens are generated on real records, so text the
design writes into its html stays fixed. The design and modification prompts
forbid model paths, invented values and promised behaviour in that text.
"""

from diagram_handlers.types.gui_nocode_diagram_handler import GUINoCodeDiagramHandler
from schemas import AuthoredSystemGUISpec
from schemas.gui_diagram import GUIModificationBatchSpec

ORDER_METADATA = [
    {
        "id": "cls-order",
        "name": "Order",
        "attributes": [
            {"id": "o-total", "name": "total", "type": "float", "isNumeric": True, "isString": False},
        ],
        "methods": [],
    },
]

SECTION = "<section class='app-head'><h1>Orders</h1><p>Every order of the shop.</p></section>"


def _generate(sections, class_metadata=ORDER_METADATA, capture=None):
    handler = GUINoCodeDiagramHandler.__new__(GUINoCodeDiagramHandler)

    def _predict(**kw):
        if capture is not None:
            capture.update(kw)
        return AuthoredSystemGUISpec(projectName="Shop", pages=[{"name": "Orders", "sections": sections}])

    handler.predict_two_pass_structured = _predict
    return handler.generate_complete_system("order desk", class_metadata=class_metadata)["model"]


def test_design_prompt_states_the_static_copy_rules_only_with_a_class_diagram():
    with_classes, without = {}, {}
    _generate([{"html": SECTION}], capture=with_classes)
    _generate([{"html": SECTION}], class_metadata=None, capture=without)
    assert "STATIC COPY" in with_classes["system_prompt"]
    assert "STATIC COPY" not in without["system_prompt"]


def test_modification_prompt_states_the_static_copy_rules():
    handler = GUINoCodeDiagramHandler.__new__(GUINoCodeDiagramHandler)
    model = _generate([{"html": SECTION}])
    prompts = []

    def _predict(*args, **kwargs):
        prompts.append(kwargs["system_prompt"])
        return GUIModificationBatchSpec(operations=[{
            "operation": "append_section", "pageName": "Orders",
            "section": {"html": "<section class='s'><h2>Notes</h2><p>Kept.</p></section>"},
        }])

    handler.predict_structured = _predict
    handler.generate_modification("add notes", model, raw_request="add notes", class_metadata=ORDER_METADATA)
    assert "STATIC COPY" in prompts[0]
