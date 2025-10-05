#!/usr/bin/env bash
set -euo pipefail

# Ensure build tooling is present
python -m pip install --upgrade build twine >/dev/null 2>&1 || true

# Clean previous artifacts
rm -rf dist build *.egg-info || true

# Build sdist and wheel
python -m build

# Checksums and integrity check
( cd dist && sha256sum * > SHA256SUMS )

# Only check sdist and wheel
python -m twine check dist/*.whl dist/*.tar.gz

# Print summary
echo "\nArtifacts:"
ls -lh dist
