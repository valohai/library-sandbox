# library-sandbox

This is a beta sandbox of reusable steps for the Valohai ecosystem.

## Development

### Linting

Linting/formatting happens via `pre-commit`. Install it with `pip install pre-commit` and:

- run `pre-commit install` to install Git hooks if you like, or
- run `pre-commit run` manually to run everything on staged files, or
- `pre-commit run --all-files` to run everything on all files.

The linters run by `pre-commit` are `ruff`, `black`, and `prettier`;
you can (should) set up your IDE to run them automatically too.
