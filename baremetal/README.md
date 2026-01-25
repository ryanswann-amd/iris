# Baremetal Testing Backend

This directory contains scripts for running Iris tests in a baremetal environment using Python virtual environments (venv) for isolation, rather than Docker or Apptainer containers.

## Purpose

The baremetal testing backend provides:
- Lightweight testing environment without containerization overhead
- Python venv-based isolation similar to Docker/Apptainer
- Same testing scripts and workflows as container-based backends
- Automatic fallback when Docker/Apptainer are not available

## Structure

- `build.sh` - Creates and sets up the Python virtual environment
- `run.sh` - Activates the venv and executes commands
- `venv/` - Python virtual environment (auto-generated, not in git)

## Usage

### Build Environment

```bash
# Create and setup the virtual environment
bash baremetal/build.sh
```

This script will:
1. Create a Python virtual environment at `baremetal/venv/`
2. Upgrade pip to the latest version
3. Install base dependencies (wheel, jupyter)

### Run Commands

```bash
# Run a command in the baremetal environment
bash baremetal/run.sh <command>

# Examples:
bash baremetal/run.sh python --version
bash baremetal/run.sh python -c "import torch; print(torch.__version__)"
```

### Interactive Shell

```bash
# Start an interactive bash session with venv activated
bash baremetal/run.sh
```

## Integration with Testing Infrastructure

The baremetal backend integrates seamlessly with the existing testing infrastructure:

- `.github/scripts/container_build.sh` - Detects and builds baremetal environment
- `.github/scripts/container_exec.sh` - Executes commands in baremetal venv
- `.github/scripts/container_run.sh` - Runs interactive sessions
- `.github/scripts/run_tests.sh` - Works with baremetal backend

### Auto-Detection

By default, the scripts auto-detect which backend to use:
1. Apptainer (if available)
2. Docker (if available)
3. Baremetal (fallback)

### Override Runtime Selection

You can override the auto-detection by setting the `CONTAINER_RUNTIME` environment variable:

```bash
# Force use of baremetal backend
export CONTAINER_RUNTIME=baremetal
bash .github/scripts/container_build.sh

# Or for a single command
CONTAINER_RUNTIME=baremetal bash .github/scripts/container_exec.sh "pytest tests/"
```

In CI workflows, set it in the environment variables:

```yaml
env:
  CONTAINER_RUNTIME: baremetal  # Options: apptainer, docker, baremetal
```

## Notes

- The `venv/` directory is excluded from version control via `.gitignore`
- The baremetal backend uses the host system's ROCm/HIP installation
- GPU device selection works via the `HIP_VISIBLE_DEVICES` environment variable
- No containerization overhead, but also no isolation from host system
