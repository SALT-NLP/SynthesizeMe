# Contributing to SynthesizeMe

First off, thank you for considering contributing to SynthesizeMe! We welcome any and all contributions, from bug reports to feature requests and code changes.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please open an issue on our [GitHub issue tracker](https://github.com/SaltNLP/SynthesizeMe/issues). Please include a clear title and a detailed description of the bug, including steps to reproduce it if possible.

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one, please open an issue on our [GitHub issue tracker](https://github.com/SALTNLP/SynthesizeMe/issues). Please provide a clear and detailed explanation of the feature and its potential benefits.

### Pull Requests

If you have a bug fix or a new feature that you would like to contribute, please submit a pull request.

## Development Setup

To get started with development, please follow these steps:

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/SynthesizeMe.git
    cd SynthesizeMe
    ```
3.  **Create a virtual environment**. One option is using `venv`:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
4.  **Install the package in editable mode** along with development dependencies:
    ```bash
    pip install -e .
    pip install pytest ruff
    ```

## Running Tests

We (will) use `pytest` for testing. To run the tests, simply run the following command from the root of the repository:

```bash
pytest
```

Please ensure that all tests pass before submitting a pull request.

## Code Style and Linting

We use `ruff` for code formatting and linting to maintain a consistent code style.

*   To **format** your code, run:
    ```bash
    ruff format .
    ```
*   To **check for linting errors** and automatically fix them when possible, run:
    ```bash
    ruff check . --fix
    ```

Please make sure your code is formatted and passes the linter checks before submitting a pull request.

## Submitting a Pull Request

1.  Create a new branch for your changes:
    ```bash
    git checkout -b your-branch-name
    ```
2.  Make your changes and commit them with a clear and concise commit message.
3.  Push your changes to your fork on GitHub:
    ```bash
    git push origin your-branch-name
    ```
4.  Open a pull request from your fork to the `main` branch of the official SynthesizeMe repository.
5.  In the pull request description, please provide a detailed explanation of your changes and reference any related issues.

Thank you again for your contribution! 