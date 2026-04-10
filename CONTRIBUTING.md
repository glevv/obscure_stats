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
- This project uses `poetry` as a pakcage manager. See how to install it [here](https://python-poetry.org/docs/#installation).
- Set up local environment that `poetry` will use. You can do it with [pyenv](https://github.com/pyenv/pyenv#installation) or venv or any other environment manager that you like.
- to initialize your local environment run:
```bash
>>> poetry install
```
- You are good to go!

## Workflow

- Every change should be tested; you need to add new tests for the new functionality (`pytest` and `pytest-cov` will help you with this).
- Every change should be documented; you need to add a docstring (`numpy` style) with reference to a scientific paper (preprints accepted).
- Every change should be clean; you need to run linters, formatters, type checkers (`ruff` and `mypy` will take care of this).

After you have made some changes to the codebase, you should run the following commands:
```python
>>> poetry run ruff check . --fix
```
This command will run linters and other useful stuff and try to fix all the problems. If something is unfixable automatically, you should try to fix it manually.

```python
>>> poetry run ruff format .
```
This command will run formatter.

```python
>>> poetry run mypy .
```
This command will run type checker. All typing problems should be fixed.

```python
>>> poetry run pytest --cov-report term-missing --cov=obscure_stats
```
This command will run the test suite. All tests should pass, as well as code coverage should be high enough.


Happy coding!
