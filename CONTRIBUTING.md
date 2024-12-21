# Contributing

We appreciate your interest in improving this project. Here are some guidelines:

1. Fork the repository and create a new branch.
2. Follow existing code styles and conventions.
3. Write clear commit messages.
4. Test your changes thoroughly.
5. Open a Pull Request with a concise summary.

## Setting Up the Development Environment

To set up the development environment, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/JaxPM.git
    cd JaxPM
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv --system-site-packages venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements-test.txt
    pip install -e .
    ```

4. **Install pre-commit hooks**:
    ```sh
    pre-commit install
    ```
    This will run code formatting and linting checks before each commit.

## Running Tests

To run the tests, use the following command:

```sh
python -m pytest
```

Make sure all tests pass before submitting your changes.

Thank you for contributing!
