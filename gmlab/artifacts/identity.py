"""Portable calculation identities and streaming payload checksums (stdlib only)."""

import hashlib

from ..utils.io import canonical_json, file_digest as file_digest


def identity_key(identity):
    """Hash a complete schema-versioned calculation contract, not a filename."""
    return hashlib.sha256(canonical_json(identity)).hexdigest()
