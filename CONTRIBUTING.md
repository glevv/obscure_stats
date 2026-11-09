# Contributing to obscure_stats

Thank you for considering contributing to this project!

All contributions are appreciated, from reporting bugs to implementing new features. This guide will try to help you take the first steps.

## Setup

- [First, fork the repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo).
- Copy forked repository to local machine:
```bash
>>> git clone https://github.com/<username>/obscure_stats.git
>>> cd obscure_stats
```
- This project uses `uv` as a package manager. See how to install it [here](https://docs.astral.sh/uv/getting-started/installation/).
- Set up local environment that `uv` will use. See how to do it [here](https://docs.astral.sh/uv/guides/install-python/).
- To initialize your local environment run:
```bash
>>> uv sync
```
- You are good to go!

## Workflow

- Every change should be tested; you need to add new tests for the new functionality (`pytest` and `pytest-cov` will help you with this).
- Every change should be documented; you need to add a docstring (`numpy` style) with reference to a scientific paper (preprints accepted).
- Every change should be clean; you need to run linters, formatters, type checkers (`ruff` and `pyrefly` will take care of this).

After you have made some changes to the codebase, you should run the following commands:
```python
>>> uv run ruff check . --fix
```
This command will run linters and other useful stuff and try to fix all the problems. If something is unfixable automatically, you should try to fix it manually.

```python
>>> uv run ruff format .
```
This command will run formatter.

```python
>>> uv run pyrefly check .
```
This command will run type checker. All typing problems should be fixed.

```python
>>> uv run pytest --cov-report term-missing --cov=obscure_stats
```
This command will run the test suite. All tests should pass, as well as code coverage should be high enough.

## Conventions

- **Raw values.** Functions return the raw statistic, without consistency (normalizing)
  constants such as MAD × 1.4826. If such a constant exists, name it in the docstring.
- **No fitting or optimisation in our own code.** Implement straightforward statistics
  (sorting, ranks, counts, quantiles, closed forms). Calling scipy functions that use
  solvers internally is fine. The one exception is
  `standard_trimmed_harrell_davis_quantile`, whose highest density interval has no closed
  form and is found with `scipy.optimize.brentq`.
- **Inputs.** Annotate array inputs as `npt.ArrayLike` and convert them once at the top of
  the function: `x = np.asarray(x, dtype=np.float64)` for numeric data, `x = np.ravel(x)`
  for categorical data.
- **Docstrings.** `numpy` style. The first line starts with a verb ("Calculate ..."). Cite
  the paper the implementation follows, and note any deliberate deviation from it.
- **No duplicates.** Before adding a function, check that it is not numerically identical
  to, or a fixed transform of, an existing one. If it is closely related, document the
  relationship in Notes.
- **Changes.** Add a line to `CHANGELOG.md` under the upcoming version.

## Releasing (maintainers)

1. Set `version` in `pyproject.toml` and `version` / `date-released` in `CITATION.cff`.
   In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` and add a new
   empty `## [Unreleased]` section above it.
2. Merge to `main`, then push a tag with the plain version number:
```bash
>>> git tag 0.8.0
>>> git push origin 0.8.0
```
3. The `Release` workflow checks that the tag matches the version and that `CHANGELOG.md`
   has a dated, non-empty section for it, builds and tests the package, publishes it to
   PyPI and creates the signed GitHub release. The release text is that changelog section.

Happy coding!
