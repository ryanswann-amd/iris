#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

set -euo pipefail

# This script runs on the HOST (via devcontainer.json "initializeCommand").
# It ensures there is an ssh-agent with a stable socket at:
#   ~/.ssh/ssh-agent.sock
#
# It also tries to load ~/.ssh/id_rsa if present.
# If your key is passphrase-protected and you're non-interactive, it may fail silently.

SOCK="${HOME}/.ssh/ssh-agent.sock"

mkdir -p "${HOME}/.ssh"

# Check if socket exists and is working
if [[ -S "${SOCK}" ]]; then
  # Verify the agent is responsive and has keys
  if SSH_AUTH_SOCK="${SOCK}" ssh-add -l >/dev/null 2>&1; then
    # Agent is running and has keys loaded, we're done
    exit 0
  fi
fi

# Either socket doesn't exist, or agent has no keys
# Remove stale socket if present
rm -f "${SOCK}"

# Start a new ssh-agent
ssh-agent -a "${SOCK}" -t 8h >/dev/null

# Try to load default keys
if [[ -f "${HOME}/.ssh/id_rsa" ]]; then
  SSH_AUTH_SOCK="${SOCK}" ssh-add "${HOME}/.ssh/id_rsa" >/dev/null 2>&1 || true
fi

# List loaded keys (for verification, ignores errors)
SSH_AUTH_SOCK="${SOCK}" ssh-add -l >/dev/null 2>&1 || true
