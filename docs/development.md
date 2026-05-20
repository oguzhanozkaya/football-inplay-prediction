---
description: Conventions, rules and policies for MoonAI development.
---

# Development

## Standards

### Python

#### Toolchain

- **uv**: 0.11+
- **Python**: 3.14+

#### Code Style

- Quote style: double
- Indent style: space

## Workflow

### Development Commands

- `just sync` installs the Python environment.
- `just run` runs the main pipeline.
- `just docs` serves the documentation site locally.

### Quality and Verification Commands

- `just check` runs formatting and lint checks.
- `just fix` runs the automated format and lint fixes.
