"""Error code taxonomy for the sleap-connect protocol v1 (spec §8).

Every `res` error is ``{code, msg, data?}``. `code` is a dotted, namespaced,
machine-matchable string a client can branch on, instead of pattern-matching
human-readable text — replacing today's ad-hoc error strings like
``FS_ERROR::http_error::...``.
"""

from typing import Any, Optional

# auth.*
AUTH_REQUIRED = "auth.required"
AUTH_UNTRUSTED = "auth.untrusted"
AUTH_BAD_SIGNATURE = "auth.bad_signature"
AUTH_PAIRING_EXPIRED = "auth.pairing_expired"

# proto.*
PROTO_MISMATCH = "proto.mismatch"
PROTO_UNKNOWN_METHOD = "proto.unknown_method"

# fs.*
FS_NOT_FOUND = "fs.not_found"
FS_FORBIDDEN = "fs.forbidden"
FS_IO_ERROR = "fs.io_error"

# blob.*
BLOB_UNKNOWN = "blob.unknown"
BLOB_INCOMPLETE = "blob.incomplete"
BLOB_HASH_MISMATCH = "blob.hash_mismatch"

# job.*
JOB_NOT_FOUND = "job.not_found"
JOB_ALREADY_TERMINAL = "job.already_terminal"
JOB_SPEC_INVALID = "job.spec_invalid"
JOB_ACTIVE = "job.active"

# Catch-all for an unexpected worker-side fault.
INTERNAL = "internal"


class ProtocolError(Exception):
    """An error that maps directly onto a `res.error` frame.

    Raise this from a method handler to control exactly what the client
    sees; any other exception is caught by the server and reported as
    `INTERNAL` with the exception's message (see `server.ProtocolV1Server._
    dispatch`).
    """

    def __init__(self, code: str, msg: str, data: Optional[Any] = None):
        """Initialize the error.

        Args:
            code: One of the namespaced codes above (or a new one following
                the same ``namespace.reason`` convention).
            msg: Human-readable message, for logs.
            data: Optional structured context.
        """
        super().__init__(msg)
        self.code = code
        self.data = data
