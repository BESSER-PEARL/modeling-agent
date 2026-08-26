"""Meta / value-proposition questions must be ANSWERED, not routed into a build
or a generic greeting. "do you also generate websites?" used to kick off a full
system; "why use you instead of Claude/GPT?" used to get the generic greeting.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from state_bodies import (  # noqa: E402
    _meta_question_answer as _meta,
    _VALUE_PROP_ANSWER,
    _CAPABILITY_ANSWER,
)


def test_value_prop_questions():
    for msg in [
        "why should I use you instead of claude or gpt?",
        "why use you?",
        "why would I use you rather than chatgpt",
        "what makes you different?",
        "why besser?",
        "you vs claude",
        "why not just use gpt",
    ]:
        assert _meta(msg) == _VALUE_PROP_ANSWER, msg


def test_capability_questions():
    for msg in [
        "do you also generate websites?",
        "do you generate code?",
        "do you support django?",
        "do you build apps?",
        "do you make a rest api?",
    ]:
        assert _meta(msg) == _CAPABILITY_ANSWER, msg


def test_real_build_requests_are_not_meta():
    for msg in [
        "create a library management system",
        "generate a rest api",
        "build me a todo app",
        "can you build me a shop app",          # a build request, not "do you…?"
        "make a website for a bakery",           # a build request
        "why is my model not generating code",   # a support question, not value-prop
        "add a Priority enum to my model",
    ]:
        assert _meta(msg) is None, msg
