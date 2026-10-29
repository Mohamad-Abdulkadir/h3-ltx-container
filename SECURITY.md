# Security

Report vulnerabilities privately through the repository owner's GitHub security
advisory channel when enabled. Do not include tokens, private media or full shell
environments in public issues. Rotate any token accidentally exposed.

Expose only the authenticated runtime API through an HTTPS proxy. ComfyUI and its
custom-node routes must remain loopback-only. Use a 32-character-or-longer API token
from a secure source, environment secrets, and public-key-only SSH if needed.
Treat model files and custom nodes as executable-supply-chain inputs: pinned hashes
reduce substitution risk but do not establish safety. The optional YOLO detector is
a pickle-based PyTorch file and must be explicitly reviewed before population.

URL uploads reject private addresses and pin validated public IPs through redirects.
Archive extraction rejects traversal and symlinks. These controls are tested locally;
independent security review and deployment-level access controls remain recommended.
The API is a single-operator service, without tenant isolation or per-token quotas.

No runtime deletion occurs automatically. Run-record pruning is an explicit admin CLI
with a dry-run default. Back up the SQLite database together with the run directory;
use SQLite's backup API or stop the runtime for a consistent filesystem backup.
