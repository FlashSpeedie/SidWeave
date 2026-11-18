# Contributing to SidWeave

Thank you for your interest in contributing to SidWeave! This document outlines the process for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Issues

1. Check if the issue already exists in the issue tracker
2. Create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce (if applicable)
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Relevant logs or error messages

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `pytest`
5. Run linting: `ruff check .`
6. Run type checking: `mypy src/`
7. Commit with clear messages
8. Push to your fork
9. Open a pull request

### Pull Request Guidelines

- Keep PRs focused and small
- Include tests for new functionality
- Update documentation as needed
- Ensure all checks pass
- Respond to review feedback promptly

## Development Setup

```bash
# Clone the repository
git clone https://github.com/SidEraLabs/SidWeave.git
cd SidWeave

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy src/
```

## Code Style

- Follow PEP 8 (enforced by ruff)
- Use type hints (enforced by mypy)
- Write clear, descriptive names
- Keep functions small and focused
- Add docstrings for public APIs
- Minimal comments - code should be self-documenting

## Testing

- Write unit tests for new functionality
- Aim for >80% coverage
- Use pytest fixtures for shared setup
- Mock external dependencies

## Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Keep documentation honest - don't advertise unimplemented features

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release tag
4. Build and publish to PyPI

## Questions?

Open an issue or start a discussion. We're happy to help!