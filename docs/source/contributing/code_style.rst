Code Style & Conventions
========================

This section describes the coding standards, naming conventions, and design
principles followed in the Modeling Agent codebase.

.. contents:: On this page
   :local:
   :depth: 2

General Rules
-------------

- **Imports**: Standard library → third-party → local, separated by blank lines
- **Type hints**: Use for function signatures; skip for obvious local variables
- **Error handling**: Only at system boundaries (user input, LLM responses,
  external APIs). Trust internal code.
- **Naming**: ``snake_case`` for functions/variables, ``PascalCase`` for classes,
  ``UPPER_CASE`` for constants
- **Comments**: Only where the logic isn't self-evident. No docstrings for
  obvious methods.
- **No premature abstraction**: Three similar lines is better than a premature
  helper function


Design Principles
-----------------

Minimal Complexity
~~~~~~~~~~~~~~~~~~

The right amount of complexity is the minimum needed for the current task.
Don't add error handling, fallbacks, or validation for scenarios that can't
happen. Don't design for hypothetical future requirements.

No Unnecessary Features
~~~~~~~~~~~~~~~~~~~~~~~

Don't add features, refactor code, or make "improvements" beyond what was
asked. A bug fix doesn't need surrounding code cleaned up. A simple feature
doesn't need extra configurability.

Trust Internal Code
~~~~~~~~~~~~~~~~~~~

Only validate at system boundaries (user input, external APIs). Don't add
defensive checks for internal function calls that are guaranteed to produce
valid data.


Commit Message Format
---------------------

Use `conventional commits <https://www.conventionalcommits.org/>`_:

- ``fix:`` — bug fix
- ``feat:`` — new feature
- ``refactor:`` — code restructuring without behavior change
- ``docs:`` — documentation only
- ``test:`` — adding or modifying tests

Example:

.. code-block:: text

   fix: prevent "generate a class diagram" from misrouting to code generation

   The LLM classifier sometimes classifies "generate a class diagram" as
   generation_intent. Added pre-filter guard and cross-validation to catch
   this case.
