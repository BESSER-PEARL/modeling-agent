"""Out-of-scope guard: a request to PRODUCE a non-software artifact (an actual
image/picture, creative writing, a joke) must be redirected — the assistant
models software systems and generates code, it does not draw cats or write
poems. HIGH-PRECISION: a real modeling request whose DOMAIN involves these
('model a photo-sharing app', 'a story management system') must NOT be caught.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from state_bodies import _request_is_out_of_scope as _oos  # noqa: E402


def test_off_topic_artifact_requests_are_caught():
    for msg in [
        "generate a picture of a cat",
        "draw me an image of a sunset",
        "create a picture of a house",
        "make a logo of a dragon",
        "render an illustration of a robot",
        "paint a portrait of a queen",
        "give me an avatar of a wizard",
        "write me a poem about the sea",
        "compose a song about summer",
        "write a short story about robots",
        "make a joke about databases",
        "give me a haiku",
        "tell me a joke",
        "tell me a story",
    ]:
        assert _oos(msg), f"expected out-of-scope: {msg!r}"


def test_real_modeling_requests_are_not_caught():
    for msg in [
        "model a photo-sharing app",
        "design a system for a poetry contest",
        "create a CRM for an art gallery",
        "create a library management system",
        "generate a picture management system",     # 'picture' but no 'of'
        "build a story management app",             # 'story' + software noun
        "create a song request system",            # 'song' + software noun
        "draw the class diagram for a bank",         # 'diagram' not an artifact word
        "create a diagram of a shop",               # 'diagram' not in the image list
        "generate django code for my model",
        "write a python class for User",            # 'python class' not creative
        "add an image attribute to the Product class",
    ]:
        assert not _oos(msg), f"expected NOT out-of-scope: {msg!r}"
