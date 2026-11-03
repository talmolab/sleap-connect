"""Jobs module for sleap-connect.

This module provides structured job specifications for training and inference:
- spec: Job specification data types (TrainJobSpec, TrackJobSpec)
- validator: Validation of job specs against filesystem and constraints
- builder: Command building from validated job specs
"""

from sleap_connect.jobs.spec import (
    TrainJobSpec,
    TrackJobSpec,
    parse_job_spec,
)
from sleap_connect.jobs.validator import (
    JobValidator,
    ValidationError,
    NUMERIC_CONSTRAINTS,
)
from sleap_connect.jobs.builder import (
    CommandBuilder,
    DEFAULT_ZMQ_PORTS,
)

__all__ = [
    # Spec types
    "TrainJobSpec",
    "TrackJobSpec",
    "parse_job_spec",
    # Validation
    "JobValidator",
    "ValidationError",
    "NUMERIC_CONSTRAINTS",
    # Command building
    "CommandBuilder",
    "DEFAULT_ZMQ_PORTS",
]
