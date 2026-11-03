"""Job specification validation.

This module provides validation for job specifications, checking:
- Paths are within allowed mounts
- Paths exist on the filesystem
- Numeric values are within valid ranges
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import yaml

from sleap_connect.jobs.spec import TrainJobSpec, TrackJobSpec


@dataclass
class ValidationError:
    """A validation error for a job specification field.

    Attributes:
        field: Name of the field that failed validation
        message: Human-readable error message
        code: Error code for programmatic handling (e.g., PATH_NOT_FOUND)
        path: The invalid path (if path-related error)
    """

    field: str
    message: str
    code: Optional[str] = None
    path: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {"field": self.field, "message": self.message}
        if self.code is not None:
            result["code"] = self.code
        if self.path is not None:
            result["path"] = self.path
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ValidationError":
        """Create from dictionary."""
        return cls(
            field=data["field"],
            message=data["message"],
            code=data.get("code"),
            path=data.get("path"),
        )


# Numeric constraints for job parameters
NUMERIC_CONSTRAINTS = {
    "max_epochs": {"min": 1, "max": 10000},
    "batch_size": {"min": 1, "max": 256},
    "learning_rate": {"min": 1e-10, "max": 1.0},
    "peak_threshold": {"min": 0.0, "max": 1.0},
}


class JobValidator:
    """Validates job specifications before execution.

    The validator checks that:
    - All paths are within allowed mount points
    - All paths that should exist do exist
    - Numeric values are within valid ranges

    Attributes:
        file_manager: FileManager instance for path validation (optional)
        mounts: List of mount configurations (used if file_manager not provided)
    """

    def __init__(
        self,
        file_manager=None,
        mounts: Optional[List] = None,
    ):
        """Initialize validator.

        Args:
            file_manager: FileManager instance (has mounts and _is_path_allowed)
            mounts: List of MountConfig objects (alternative to file_manager)

        Note:
            Either file_manager or mounts must be provided. If both are provided,
            file_manager takes precedence.
        """
        self.file_manager = file_manager
        self._mounts = mounts

    @property
    def mounts(self) -> List:
        """Get the list of mounts."""
        if self.file_manager is not None:
            return self.file_manager.mounts
        return self._mounts or []

    def validate_train_spec(self, spec: TrainJobSpec) -> List[ValidationError]:
        """Validate a training job specification.

        Args:
            spec: TrainJobSpec to validate

        Returns:
            List of ValidationError objects (empty if valid)
        """
        errors = []

        # Validate config source: either config_paths or config_content required
        valid_config_indices = []
        has_config_content = getattr(spec, "config_content", None) is not None
        if not spec.config_paths and not has_config_content:
            errors.append(
                ValidationError(
                    field="config",
                    message="Either config_paths or config_content is required",
                    code="MISSING_CONFIG",
                )
            )
        elif spec.config_paths:
            for i, config_path in enumerate(spec.config_paths):
                # Use indexed field name for multiple configs
                field_name = (
                    f"config_path[{i}]" if len(spec.config_paths) > 1 else "config_path"
                )
                error = self._validate_path(config_path, field_name, must_exist=True)
                if error:
                    errors.append(error)
                else:
                    valid_config_indices.append(i)

        # Validate internal paths in config files (only for configs that exist)
        for i in valid_config_indices:
            config_errors = self._validate_config_internals(
                spec.config_paths[i], spec, config_index=i
            )
            errors.extend(config_errors)

        # Validate labels path if provided
        if spec.labels_path:
            error = self._validate_path(
                spec.labels_path, "labels_path", must_exist=True
            )
            if error:
                errors.append(error)

        # Validate val_labels path if provided
        if spec.val_labels_path:
            error = self._validate_path(
                spec.val_labels_path, "val_labels_path", must_exist=True
            )
            if error:
                errors.append(error)

        # Validate resume checkpoint if provided
        if spec.resume_ckpt_path:
            error = self._validate_path(
                spec.resume_ckpt_path, "resume_ckpt_path", must_exist=True
            )
            if error:
                errors.append(error)

        # Validate numeric fields
        if spec.max_epochs is not None:
            error = self._validate_numeric("max_epochs", spec.max_epochs)
            if error:
                errors.append(error)

        if spec.batch_size is not None:
            error = self._validate_numeric("batch_size", spec.batch_size)
            if error:
                errors.append(error)

        if spec.learning_rate is not None:
            error = self._validate_numeric("learning_rate", spec.learning_rate)
            if error:
                errors.append(error)

        return errors

    def validate_track_spec(self, spec: TrackJobSpec) -> List[ValidationError]:
        """Validate a tracking/inference job specification.

        Args:
            spec: TrackJobSpec to validate

        Returns:
            List of ValidationError objects (empty if valid)
        """
        errors = []

        # Validate data path (required)
        error = self._validate_path(spec.data_path, "data_path", must_exist=True)
        if error:
            errors.append(error)

        # Validate model paths (required, must be non-empty)
        if not spec.model_paths:
            errors.append(
                ValidationError(
                    field="model_paths",
                    message="At least one model path is required",
                )
            )
        else:
            for i, model_path in enumerate(spec.model_paths):
                error = self._validate_path(
                    model_path, f"model_paths[{i}]", must_exist=True
                )
                if error:
                    errors.append(error)

        # Validate output path directory if provided
        if spec.output_path:
            parent = str(Path(spec.output_path).parent)
            error = self._validate_path(
                parent, "output_path (parent directory)", must_exist=True
            )
            if error:
                errors.append(error)

        # Validate numeric fields
        if spec.batch_size is not None:
            error = self._validate_numeric("batch_size", spec.batch_size)
            if error:
                errors.append(error)

        if spec.peak_threshold is not None:
            error = self._validate_numeric("peak_threshold", spec.peak_threshold)
            if error:
                errors.append(error)

        return errors

    def validate(
        self, spec: Union[TrainJobSpec, TrackJobSpec]
    ) -> List[ValidationError]:
        """Validate any job specification.

        Args:
            spec: TrainJobSpec or TrackJobSpec to validate

        Returns:
            List of ValidationError objects (empty if valid)
        """
        if isinstance(spec, TrainJobSpec):
            return self.validate_train_spec(spec)
        elif isinstance(spec, TrackJobSpec):
            return self.validate_track_spec(spec)
        else:
            return [
                ValidationError(
                    field="type",
                    message=f"Unknown job spec type: {type(spec).__name__}",
                )
            ]

    def _validate_path(
        self, path: str, field: str, must_exist: bool = True
    ) -> Optional[ValidationError]:
        """Validate a path is within allowed mounts and optionally exists.

        Args:
            path: Path string to validate
            field: Field name for error reporting
            must_exist: Whether the path must exist

        Returns:
            ValidationError if invalid, None if valid
        """
        try:
            resolved = Path(path).resolve()
        except (OSError, ValueError) as e:
            return ValidationError(
                field, f"Invalid path: {e}", code="INVALID_PATH", path=path
            )

        # Check path is within allowed mounts
        if not self._is_path_allowed(resolved):
            return ValidationError(
                field, "Path not within allowed mounts", code="NOT_ALLOWED", path=path
            )

        # Check path exists
        if must_exist and not resolved.exists():
            return ValidationError(
                field, "Path does not exist", code="PATH_NOT_FOUND", path=path
            )

        return None

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if a path is within an allowed mount.

        Uses FileManager's method if available, otherwise checks mounts directly.

        Args:
            path: Path to check

        Returns:
            True if path is within a configured mount
        """
        # Use FileManager's method if available
        if self.file_manager is not None:
            return self.file_manager._is_path_allowed(path)

        # Otherwise, check mounts directly
        if not self.mounts:
            return False

        try:
            resolved = path.resolve()
        except (OSError, ValueError):
            return False

        for mount in self.mounts:
            mount_path = Path(mount.path).resolve()
            try:
                resolved.relative_to(mount_path)
                return True
            except ValueError:
                continue

        return False

    def _validate_numeric(
        self, field: str, value: Union[int, float]
    ) -> Optional[ValidationError]:
        """Validate a numeric value is within allowed range.

        Args:
            field: Field name for error reporting
            value: Numeric value to validate

        Returns:
            ValidationError if invalid, None if valid
        """
        if field not in NUMERIC_CONSTRAINTS:
            return None

        constraints = NUMERIC_CONSTRAINTS[field]
        min_val = constraints["min"]
        max_val = constraints["max"]

        if value < min_val or value > max_val:
            return ValidationError(
                field,
                f"Value must be between {min_val} and {max_val}, got {value}",
            )

        return None

    def _validate_config_internals(
        self, config_path: str, spec: TrainJobSpec, config_index: int = 0
    ) -> List[ValidationError]:
        """Validate paths inside a config YAML file.

        Parses the config file and validates data_config.train_labels_path
        and data_config.val_labels_path if they are not overridden by
        the job spec's labels_path and val_labels_path.

        Args:
            config_path: Path to the config YAML file
            spec: TrainJobSpec containing potential overrides
            config_index: Index of the config (for error field naming)

        Returns:
            List of ValidationError objects for invalid internal paths
        """
        errors = []
        config_prefix = (
            f"config[{config_index}]." if len(spec.config_paths) > 1 else "config."
        )

        # Try to parse the config file
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            errors.append(
                ValidationError(
                    field=f"{config_prefix}parse",
                    message=f"Failed to parse config file: {e}",
                    code="CONFIG_PARSE_ERROR",
                    path=config_path,
                )
            )
            return errors
        except OSError as e:
            # File doesn't exist or can't be read - already caught by path validation
            return errors

        if not isinstance(config, dict):
            errors.append(
                ValidationError(
                    field=f"{config_prefix}parse",
                    message="Config file is not a valid YAML mapping",
                    code="CONFIG_PARSE_ERROR",
                    path=config_path,
                )
            )
            return errors

        # Get data_config section
        data_config = config.get("data_config", {})
        if not isinstance(data_config, dict):
            data_config = {}

        # Validate train_labels_path if not overridden by spec.labels_path
        if not spec.labels_path:
            train_labels = data_config.get("train_labels_path")
            if train_labels is not None:
                # sleap-nn accepts both string and list for train_labels_path
                if isinstance(train_labels, str):
                    error = self._validate_path(
                        train_labels,
                        f"{config_prefix}train_labels_path",
                        must_exist=True,
                    )
                    if error:
                        errors.append(error)
                elif isinstance(train_labels, list):
                    for i, path in enumerate(train_labels):
                        if isinstance(path, str):
                            field_name = f"{config_prefix}train_labels_path[{i}]"
                            error = self._validate_path(
                                path, field_name, must_exist=True
                            )
                            if error:
                                errors.append(error)

        # Validate val_labels_path if not overridden by spec.val_labels_path
        if not spec.val_labels_path:
            val_labels = data_config.get("val_labels_path")
            if val_labels is not None:
                # sleap-nn accepts both string and list for val_labels_path
                if isinstance(val_labels, str):
                    error = self._validate_path(
                        val_labels, f"{config_prefix}val_labels_path", must_exist=True
                    )
                    if error:
                        errors.append(error)
                elif isinstance(val_labels, list):
                    for i, path in enumerate(val_labels):
                        if isinstance(path, str):
                            field_name = f"{config_prefix}val_labels_path[{i}]"
                            error = self._validate_path(
                                path, field_name, must_exist=True
                            )
                            if error:
                                errors.append(error)

        return errors
