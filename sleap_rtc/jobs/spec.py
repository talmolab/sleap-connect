"""Job specification data types for structured job submission.

This module defines the data structures for training and inference jobs
that can be serialized, validated, and converted to sleap-nn commands.
"""

from dataclasses import dataclass, asdict, field
from typing import ClassVar, Dict, Optional, List, Set, Union
import json


@dataclass
class TrainJobSpec:
    """Specification for a training job.

    Attributes:
        config_path: Full path to config YAML file (for single model training)
        config_paths: List of config paths (for multi-model training like top-down)
        config_content: Serialized training config (YAML string) sent over datachannel
        config_contents: List of serialized configs (for multi-model pipeline training)
        model_types: Model type labels matching config_contents order
            (e.g. ["centroid", "centered_instance"])
        labels_path: Override for data_config.train_labels_path
        labels_content: Base64-encoded raw .slp bytes, sent inline when the
            client has no worker-resolvable path for the labels file (same
            "send the bytes, not a path" escape hatch config_contents already
            provides for configs). The worker materializes this to a temp
            file and uses it in place of labels_path — see
            `JobMethods._materialize_labels_content`.
        val_labels_path: Override for data_config.val_labels_path
        max_epochs: Maximum training epochs
        batch_size: Batch size for training and validation
        learning_rate: Learning rate for optimizer
        run_name: Name for the training run (used in checkpoint directory)
        resume_ckpt_path: Path to checkpoint for resuming training
        path_mappings: Maps original client-side paths to resolved worker-side paths

    Note:
        Either config_path/config_paths, config_content, or config_contents
        must be provided. If config_content is provided, it is normalized to
        config_contents = [config_content]. For top-down training, provide
        both centroid and centered_instance configs in config_contents with
        matching model_types - they will be trained sequentially.
    """

    config_path: Optional[str] = None
    config_paths: List[str] = field(default_factory=list)
    config_content: Optional[str] = None
    config_contents: List[str] = field(default_factory=list)
    model_types: List[str] = field(default_factory=list)
    labels_path: Optional[str] = None
    labels_content: Optional[str] = None
    val_labels_path: Optional[str] = None
    max_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    run_name: Optional[str] = None
    resume_ckpt_path: Optional[str] = None
    path_mappings: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize config_path/config_paths after initialization."""
        # If config_path is provided but config_paths is empty, use config_path
        if self.config_path and not self.config_paths:
            self.config_paths = [self.config_path]
        # Normalize single config_content into config_contents list
        if self.config_content and not self.config_contents:
            self.config_contents = [self.config_content]
        # Ensure we have at least one config source
        if (
            not self.config_paths
            and not self.config_path
            and not self.config_content
            and not self.config_contents
        ):
            raise ValueError(
                "Must provide either config_path, config_paths, "
                "config_content, or config_contents"
            )

    def to_json(self) -> str:
        """Serialize spec to JSON string."""
        data = {"type": "train", **asdict(self)}
        # Remove None values, empty lists, and empty dicts for cleaner JSON
        data = {k: v for k, v in data.items() if v is not None and v != [] and v != {}}
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> "TrainJobSpec":
        """Deserialize spec from JSON string."""
        parsed = json.loads(data)
        parsed.pop("type", None)
        # Handle backward compatibility: config_path -> config_paths
        # Keep both fields for compatibility when coming from old format
        if "config_path" in parsed and "config_paths" not in parsed:
            parsed["config_paths"] = [parsed["config_path"]]
        # Filter to only known fields for forward compatibility
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        parsed = {k: v for k, v in parsed.items() if k in known_fields}
        return cls(**parsed)

    def to_dict(self) -> dict:
        """Convert spec to dictionary with type field."""
        data = {"type": "train", **asdict(self)}
        return {k: v for k, v in data.items() if v is not None and v != [] and v != {}}

    @classmethod
    def from_dict(cls, data: dict) -> "TrainJobSpec":
        """Create spec from dictionary."""
        data = dict(data)  # Make a copy
        data.pop("type", None)
        # Filter to only known fields for forward compatibility
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**data)


@dataclass
class TrackJobSpec:
    """Specification for an inference/tracking job.

    Attributes:
        data_path: Path to .slp file or video for inference (required)
        model_paths: Paths to model directories (required, can be multiple)
        output_path: Path for output predictions file
        batch_size: Batch size for inference
        peak_threshold: Peak detection threshold (0.0-1.0)
        only_suggested_frames: Deprecated; use frame_filter="suggested" instead.
            Kept for backward compatibility — when True and frame_filter is None,
            it auto-migrates to frame_filter="suggested" in __post_init__.
        frames: Frame range string (e.g., "0-100,200-300")
        frame_filter: Which subset of frames to run inference on. One of
            "suggested", "user", "predicted", "random", or None (all frames).
        video_index: Restrict inference to a single video by index. None = all
            videos in the labels file.
    """

    data_path: str
    model_paths: List[str] = field(default_factory=list)
    output_path: Optional[str] = None
    batch_size: Optional[int] = None
    peak_threshold: Optional[float] = None
    only_suggested_frames: bool = False
    exclude_user_labeled: bool = False
    frames: Optional[str] = None
    frame_filter: Optional[str] = None
    video_index: Optional[int] = None
    path_mappings: Dict[str, str] = field(default_factory=dict)
    # Tracking parameters
    tracker: Optional[str] = None  # "simple" or "flow"
    similarity: Optional[str] = None  # "oks", "iou", "centroids", "euclidean_dist"
    match: Optional[str] = None  # "hungarian" or "greedy"
    track_window: Optional[int] = None
    robust: Optional[float] = None
    max_tracks: Optional[int] = None
    connect_single_breaks: bool = False
    # Preprocessing
    ensure_channels: Optional[str] = None  # "rgb" or "grayscale"

    _VALID_FRAME_FILTERS: ClassVar[Set[Optional[str]]] = {
        None,
        "suggested",
        "user",
        "predicted",
        "random",
    }

    def __post_init__(self):
        """Migrate deprecated only_suggested_frames flag and validate frame_filter."""
        # Backward-compat migration: only_suggested_frames=True -> frame_filter="suggested"
        # Only migrate if frame_filter wasn't explicitly set, so explicit values win.
        if self.only_suggested_frames and self.frame_filter is None:
            self.frame_filter = "suggested"
        # Validate frame_filter (after migration so the migrated value is checked).
        if self.frame_filter not in self._VALID_FRAME_FILTERS:
            valid = sorted(v for v in self._VALID_FRAME_FILTERS if v is not None)
            raise ValueError(
                f"frame_filter must be one of {valid} or None; "
                f"got {self.frame_filter!r}"
            )

    def to_json(self) -> str:
        """Serialize spec to JSON string."""
        data = {"type": "track", **asdict(self)}
        # Remove None values and False booleans for cleaner JSON
        data = {
            k: v
            for k, v in data.items()
            if v is not None and v is not False and v != []
        }
        # Always include model_paths even if empty (it's required)
        if "model_paths" not in data:
            data["model_paths"] = self.model_paths
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> "TrackJobSpec":
        """Deserialize spec from JSON string."""
        parsed = json.loads(data)
        parsed.pop("type", None)
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        parsed = {k: v for k, v in parsed.items() if k in known_fields}
        return cls(**parsed)

    def to_dict(self) -> dict:
        """Convert spec to dictionary with type field."""
        data = {"type": "track", **asdict(self)}
        return {
            k: v
            for k, v in data.items()
            if v is not None and v is not False and v != []
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrackJobSpec":
        """Create spec from dictionary."""
        data = dict(data)  # Make a copy
        data.pop("type", None)
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**data)


def parse_job_spec(json_str: str) -> Union[TrainJobSpec, TrackJobSpec]:
    """Parse a JSON string into the appropriate job spec type.

    Args:
        json_str: JSON string with "type" field indicating spec type

    Returns:
        TrainJobSpec or TrackJobSpec depending on type field

    Raises:
        ValueError: If type field is missing or unknown
    """
    parsed = json.loads(json_str)
    job_type = parsed.get("type")

    if job_type == "train":
        return TrainJobSpec.from_json(json_str)
    elif job_type == "track":
        return TrackJobSpec.from_json(json_str)
    else:
        raise ValueError(f"Unknown job type: {job_type}")
