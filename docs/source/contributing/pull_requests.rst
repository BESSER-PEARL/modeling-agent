Pull Request Process
====================

This section covers how to submit contributions, what reviewers look for, and
the project's code of conduct.

.. contents:: On this page
   :local:
   :depth: 2


Submitting a Pull Request
--------------------------

1. Create a feature branch from ``main``
2. Make your changes with tests
3. Ensure ``python -m pytest`` passes with no failures
4. Update documentation if behavior changed
5. Submit a PR using the template in ``.github/pull_request_template.md``


What Reviewers Look For
-----------------------

- **Tests cover both positive and negative cases** — every "should match" test
  needs a corresponding "should NOT match" test
- **No regressions** in existing tests
- **Intent descriptions updated** if classification behavior changed
- **Documentation updated** for user-visible changes
- **No hardcoded values** where a pattern or configuration would be more
  maintainable
- **Edge cases considered** — empty strings, ``None`` values, unusual phrasing


Review Checklist for Intent Changes
-------------------------------------

If your PR modifies intent recognition, reviewers will verify:

1. Intent descriptions in ``modeling_agent.py`` include explicit examples for
   the new behavior
2. Pre-filters in ``generation_handler.py`` are updated if needed
3. Discriminating patterns in ``workspace_orchestrator.py`` are updated if
   a new diagram type signal is added
4. Tests cover the ambiguous cases listed in :doc:`debugging`
5. The cross-validation logic in ``session_helpers.py`` still works correctly


Code of Conduct
---------------

This project follows the code of conduct defined in ``CODE_OF_CONDUCT.md``.
Please read it before contributing.
