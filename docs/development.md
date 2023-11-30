## Development

Prerequisites: [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

- Installation: `poetry install`

Activate `poetry` environment: `poetry shell`. (can skip, then prepend `poetry run` to the following commands)

- Format: `black . && isort .`
- Test: `pytest .`
- Check: `flake8 . && mypy . && black --check . && isort . --check`

### Implementation details

A step-by-step description of the algorithm can be found [here](docs/algorithm.md).

A paper describing the design of **SynDiffix**, its performance, and its anonymity properties can be found [here on ArXiv](https://arxiv.org/abs/2311.09628).

A per-dimension range is internally called an interval (and handled by the `Interval` class), in order to avoid
potential name clashes with the native Python `range` API.

### Creating a new release

First, update the project's version by editing the `[tool.poetry]` section from the `pyproject.toml` file.
Then, create a new commit with the previous change and tag it with the new version. Afterwards, push everything to GitHub:

```sh
git add pyproject.toml
git commit
git push
git tag vX.Y.Z # Replace X.Y.Z with the new version.
git push -tag
```

Once the new tag is pushed to GitHub, the release publishing workflow should start running automatically
(but, depending on the repository settings, you might also need to approve its execution manually).

If using a branch, don't forget to merge the version field change into the `main` branch.

Note that once a release has been published successfully it can't be published again, even if deleted from PyPI.
