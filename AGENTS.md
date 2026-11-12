# Global Coding Style Instructions

## Documentation and comments

When writing or modifying code:

* Add clear comments or docstrings for every function and class.
* For each function, briefly document:

  * Purpose and behavior.
  * Input arguments.
  * Output / return value.
  * Data type.
  * Tensor / array shape when applicable.
  * Important assumptions or constraints.
* For each class, briefly document:

  * Purpose.
  * Main attributes.
  * Expected input/output data.
  * Important usage notes.

## Variable comments

* Add comments for important variables
* Explain the meaning, data type, shape, and unit when applicable.
* Do not comment trivial variables if the variable name is already clear.

## Important logic comments

* Add inline comments for non-obvious logic, including:

  * algorithms,
  * formulas,
  * loss functions,
  * neural network modules,
  * tensor shape transformations,
  * coordinate transforms,
  * file I/O,
  * hardware or robot interaction code.
  * some thing else...
* For important formulas or network structures, include the formula, tensor shape, or a simple diagram when it helps readability.

## Code style

* Keep code concise, readable, and easy to review.
* Prefer explicit and straightforward logic.
* Avoid unnecessary abstraction and over-encapsulation.
* Avoid unnecessary helper functions, wrapper classes, factories, or design patterns.
* Avoid unnecessary `if/else` branches.
* Avoid unnecessary `try/except` blocks.
* Do not silently swallow exceptions.
* Prefer meaningful variable names over excessive comments.
* Keep functions reasonably short, but do not split code artificially.
* Preserve the existing project style unless it conflicts with these instructions.

## Reviewability

* Make changes minimal and localized.
* Do not rewrite unrelated code.
* When modifying existing code, briefly explain the key changes.
* Prioritize correctness, clarity, and maintainability over cleverness.
