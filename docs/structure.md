---
description: Project structure, file organization, and tooling reference.
---

# Structure

## Repository Map

```
/
├── .github/                    # GitHub workflows (currently docs deployment)
├── docs/                       # Documentation source
├── src/                        # Source tree
├── .gitattributes              # Git attributes
├── .gitignore                  # Git ignore rules
├── README.md                   # Project readme
├── justfile                    # Rust project commands
├── pyproject.toml              # Python package config
├── uv.lock                     # Python dependency lock
└── zensical.toml               # Website configuration
```

## Documentation (`docs/`)

| Path             | Purpose                                                 |
| ---------------- | ------------------------------------------------------- |
| `_assets`        | Documentation assets, extra.css, extra.js, and logo.svg |
| `index.md`       | Documentation home                                      |
| `spec.md`        | Usage guide and product specifications                  |
| `development.md` | Development rules and workflow                          |
| `structure.md`   | Repo Map, project structure, and deployment             |
| `standarts.md`   | Internal architecture                                   |
| `roadmap.md`     | Plan, status, and roadmap                               |

### Deployment

Documentation is deployed through `.github/workflows/docs.yml`. On pushes to `main`, GitHub
Pages builds the site with Zensical and publishes the generated `site/` directory.
