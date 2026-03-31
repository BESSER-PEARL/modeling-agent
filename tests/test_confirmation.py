"""Tests for the confirmation flow helper functions."""

import pytest
import sys
import os
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Stub out the besser.agent module hierarchy so that importing confirmation
# does not require the full BESSER framework to be installed.
_besser = MagicMock()
for mod_name in [
    'besser',
    'besser.agent',
    'besser.agent.core',
    'besser.agent.core.session',
]:
    sys.modules.setdefault(mod_name, _besser)

# Also stub transitive imports from confirmation.py
for mod_name in [
    'protocol',
    'protocol.adapters',
    'session_helpers',
]:
    sys.modules.setdefault(mod_name, MagicMock())

from confirmation import (
    keyword_matches,
    model_has_elements,
    REPLACE_KEYWORDS,
    KEEP_KEYWORDS,
    CANCEL_KEYWORDS,
    NEW_TAB_KEYWORDS,
)


# =============================================================================
# keyword_matches — whole-word keywords
# =============================================================================


class TestKeywordMatchesWholeWord:
    """Keywords in _WHOLE_WORD_KEYWORDS ('no', 'yes', 'add', 'keep') use \\b matching."""

    def test_no_standalone(self):
        assert keyword_matches("no", "no") is True

    def test_no_in_sentence(self):
        assert keyword_matches("no", "no thanks") is True

    def test_no_does_not_match_inside_word(self):
        assert keyword_matches("no", "nothing") is False

    def test_no_does_not_match_note(self):
        assert keyword_matches("no", "please note this") is False

    def test_no_does_not_match_another(self):
        assert keyword_matches("no", "another option") is False

    def test_no_does_not_match_innovation(self):
        assert keyword_matches("no", "innovation is key") is False

    def test_yes_standalone(self):
        assert keyword_matches("yes", "yes") is True

    def test_yes_in_sentence(self):
        assert keyword_matches("yes", "yes please") is True

    def test_yes_does_not_match_eyes(self):
        assert keyword_matches("yes", "close your eyes") is False

    def test_add_standalone(self):
        assert keyword_matches("add", "add it") is True

    def test_add_does_not_match_address(self):
        assert keyword_matches("add", "update address") is False

    def test_add_does_not_match_added(self):
        assert keyword_matches("add", "I added it") is False

    def test_keep_standalone(self):
        assert keyword_matches("keep", "keep it") is True

    def test_keep_does_not_match_keeper(self):
        assert keyword_matches("keep", "the keeper") is False

    def test_keep_does_not_match_keeping(self):
        assert keyword_matches("keep", "keeping track") is False

    def test_no_case_sensitive(self):
        # The function does not lowercase internally — the caller does.
        # "No" should not match with \bno\b because 'N' != 'n'.
        assert keyword_matches("no", "No way") is False

    def test_yes_at_end_of_sentence(self):
        assert keyword_matches("yes", "I think yes") is True

    def test_no_with_punctuation(self):
        assert keyword_matches("no", "no, I don't want that") is True


# =============================================================================
# keyword_matches — partial-match (substring) keywords
# =============================================================================


class TestKeywordMatchesPartial:
    """Keywords NOT in _WHOLE_WORD_KEYWORDS use simple substring 'in' matching."""

    def test_replace_standalone(self):
        assert keyword_matches("replace", "replace") is True

    def test_replace_in_sentence(self):
        assert keyword_matches("replace", "please replace it") is True

    def test_replace_as_substring(self):
        # Substring match: "replace" inside "replacement" matches
        assert keyword_matches("replace", "replacement needed") is True

    def test_cancel_standalone(self):
        assert keyword_matches("cancel", "cancel") is True

    def test_cancel_in_sentence(self):
        assert keyword_matches("cancel", "I want to cancel") is True

    def test_cancel_as_substring(self):
        assert keyword_matches("cancel", "cancellation") is True

    def test_overwrite_match(self):
        assert keyword_matches("overwrite", "overwrite the old model") is True

    def test_merge_match(self):
        assert keyword_matches("merge", "merge them together") is True

    def test_new_one_match(self):
        assert keyword_matches("new one", "start a new one") is True

    def test_never_mind_match(self):
        assert keyword_matches("never mind", "never mind that") is True

    def test_start_fresh_match(self):
        assert keyword_matches("start fresh", "let's start fresh") is True

    def test_no_match_for_unrelated(self):
        assert keyword_matches("replace", "hello world") is False

    def test_cancel_no_match(self):
        assert keyword_matches("cancel", "hello") is False


# =============================================================================
# model_has_elements
# =============================================================================


class TestModelHasElements:
    """Tests for model_has_elements()."""

    def test_none_returns_false(self):
        assert model_has_elements(None) is False

    def test_empty_dict_returns_false(self):
        assert model_has_elements({}) is False

    def test_dict_without_elements_key_returns_false(self):
        assert model_has_elements({"relationships": {}}) is False

    def test_dict_with_empty_elements_returns_false(self):
        assert model_has_elements({"elements": {}}) is False

    def test_dict_with_non_dict_elements_returns_false(self):
        assert model_has_elements({"elements": []}) is False
        assert model_has_elements({"elements": "not a dict"}) is False
        assert model_has_elements({"elements": 42}) is False

    def test_dict_with_populated_elements_returns_true(self):
        assert model_has_elements({
            "elements": {
                "cls-1": {"id": "cls-1", "name": "User", "type": "Class"},
            }
        }) is True

    def test_dict_with_multiple_elements_returns_true(self):
        assert model_has_elements({
            "elements": {
                "cls-1": {"id": "cls-1", "name": "User"},
                "cls-2": {"id": "cls-2", "name": "Order"},
            }
        }) is True

    def test_non_dict_input_returns_false(self):
        assert model_has_elements("not a dict") is False
        assert model_has_elements(123) is False
        assert model_has_elements([]) is False

    def test_elements_none_returns_false(self):
        assert model_has_elements({"elements": None}) is False


# =============================================================================
# Keyword lists are non-empty
# =============================================================================


class TestKeywordListsNonEmpty:
    """Ensure the keyword lists are defined and populated."""

    def test_replace_keywords_non_empty(self):
        assert isinstance(REPLACE_KEYWORDS, list)
        assert len(REPLACE_KEYWORDS) > 0

    def test_keep_keywords_non_empty(self):
        assert isinstance(KEEP_KEYWORDS, list)
        assert len(KEEP_KEYWORDS) > 0

    def test_cancel_keywords_non_empty(self):
        assert isinstance(CANCEL_KEYWORDS, list)
        assert len(CANCEL_KEYWORDS) > 0

    def test_new_tab_keywords_non_empty(self):
        assert isinstance(NEW_TAB_KEYWORDS, list)
        assert len(NEW_TAB_KEYWORDS) > 0

    def test_replace_keywords_are_strings(self):
        assert all(isinstance(k, str) for k in REPLACE_KEYWORDS)

    def test_keep_keywords_are_strings(self):
        assert all(isinstance(k, str) for k in KEEP_KEYWORDS)

    def test_cancel_keywords_are_strings(self):
        assert all(isinstance(k, str) for k in CANCEL_KEYWORDS)

    def test_new_tab_keywords_are_strings(self):
        assert all(isinstance(k, str) for k in NEW_TAB_KEYWORDS)

    def test_known_replace_keywords_present(self):
        assert "replace" in REPLACE_KEYWORDS
        assert "yes" in REPLACE_KEYWORDS

    def test_known_keep_keywords_present(self):
        assert "keep" in KEEP_KEYWORDS
        assert "no" in KEEP_KEYWORDS

    def test_known_cancel_keywords_present(self):
        assert "cancel" in CANCEL_KEYWORDS

    def test_known_new_tab_keywords_present(self):
        assert "new tab" in NEW_TAB_KEYWORDS
