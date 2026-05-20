# Specification

## Usage

### Prerequisites

| Tool   | Version / Requirement | Required              |
| ------ | --------------------- | --------------------- |
| uv     | 0.11+                 | For analysis and docs |
| Python | 3.14+                 | For analysis and docs |
| just   | any                   | Recommended           |

#### Just

[Just](https://github.com/casey/just) is a handy way to save and run project specific commands. Commands, called recipes, are stored in a file called `justfile` with syntax inspired by `make`. Recipes can be run with `just RECIPE`, and listed with `just --list`.

All of the commands needed for this project can be found and used from `justfile`. Despite being highly recommended, since Just is just a command wrapper it is not required to make this project work. Contents of the `justfile` can be used manually to standardize the commands.

```bash
# these are same
just clean
cargo clean

# clean recipe looks like this at the justfile
clean:
  cargo clean
```

### Clone the project

```bash
git clone https://github.com/oguzhanozkaya/turkish-inflation-forecasting.git
cd turkish-inflation-forecasting
```

### Sync

```bash
just sync

# manual equivalent
uv sync
```

### Run

```bash
just run

# manual equivalent
uv run main
```

## Data

## Output

