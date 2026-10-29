#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${VOLUME_ROOT:-}" ]]; then
  if [[ -d /runpod-volume ]]; then export VOLUME_ROOT=/runpod-volume;
  else export VOLUME_ROOT=/workspace; fi
fi
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  install -d -m 0700 /root/.ssh /run/sshd
  printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 0600 /root/.ssh/authorized_keys
  ssh-keygen -A
  /usr/sbin/sshd -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password
fi
# An explicit command supports CPU-pod population and diagnostics without starting engines.
if (( $# )); then exec "$@"; fi
case "${MODE:-pod}" in
  pod) exec h3ltx serve ;;
  serverless) exec python -m h3ltx.serverless ;;
  *) printf 'MODE must be pod or serverless\n' >&2; exit 2 ;;
esac
