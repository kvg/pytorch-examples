# pytorch-examples
A collection of scripts I'm writing as I learn pytorch.

Current version: 0.0.1

## Development

To do development in this codebase, the python3 development package must
be installed.

After installation the development environment can be set up by
the following commands:

    python3 -mvenv venv
    . venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .

### Linting files

    # run all linting commands
    tox -e lint

    # reformat all project files
    black src tests setup.py

    # sort imports in project files
    isort -rc src tests setup.py

    # check pep8 against all project files
    flake8 src tests setup.py

    # lint python code for common errors and codestyle issues
    pylint src

### Tests

    # run all linting and test
    tox

    # run only (fast) unit tests
    tox -e unit
    
    # run only integration tests
    tox -e integration

    # run only a single test
    # (in this case, the integration tests for `annotate`)
    tox -e singletest -- tests/integration/test_annotate.py::test_annotate

    # run only linting
    tox -e lint

Note: If you run into "module not found" errors when running tox for testing, verify the modules are listed in test-requirements.txt and delete the .tox folder to force tox to refresh dependencies.
