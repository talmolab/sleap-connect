"""Tests for job specification validation."""

import pytest
from dataclasses import dataclass
from pathlib import Path

from sleap_connect.jobs.spec import TrainJobSpec, TrackJobSpec
from sleap_connect.jobs.validator import (
    JobValidator,
    ValidationError,
    NUMERIC_CONSTRAINTS,
)


@dataclass
class MockMount:
    """Mock mount configuration for testing."""

    path: str
    label: str


class TestValidationError:
    """Tests for ValidationError dataclass."""

    def test_create_error(self):
        """Test creating a validation error."""
        error = ValidationError(
            field="config_path",
            message="Path does not exist",
            path="/invalid/path.yaml",
        )

        assert error.field == "config_path"
        assert error.message == "Path does not exist"
        assert error.path == "/invalid/path.yaml"

    def test_to_dict(self):
        """Test converting error to dictionary."""
        error = ValidationError(
            field="labels_path",
            message="Path not within allowed mounts",
            code="NOT_ALLOWED",
            path="/etc/passwd",
        )
        d = error.to_dict()

        assert d["field"] == "labels_path"
        assert d["message"] == "Path not within allowed mounts"
        assert d["code"] == "NOT_ALLOWED"
        assert d["path"] == "/etc/passwd"

    def test_to_dict_without_path(self):
        """Test converting error without path to dictionary."""
        error = ValidationError(
            field="batch_size",
            message="Value must be between 1 and 256",
        )
        d = error.to_dict()

        assert d["field"] == "batch_size"
        assert d["message"] == "Value must be between 1 and 256"
        assert "code" not in d
        assert "path" not in d

    def test_from_dict(self):
        """Test creating error from dictionary."""
        d = {
            "field": "config_path",
            "message": "Path does not exist",
            "code": "PATH_NOT_FOUND",
            "path": "/missing/file.yaml",
        }
        error = ValidationError.from_dict(d)

        assert error.field == "config_path"
        assert error.message == "Path does not exist"
        assert error.code == "PATH_NOT_FOUND"
        assert error.path == "/missing/file.yaml"


class TestJobValidatorTrainSpec:
    """Tests for validating TrainJobSpec."""

    def test_valid_spec_with_existing_paths(self, tmp_path):
        """Test validation passes with valid paths."""
        # Create test files
        config_file = tmp_path / "config.yaml"
        config_file.write_text("trainer: {}")

        labels_file = tmp_path / "labels.slp"
        labels_file.write_text("labels data")

        # Create validator with tmp_path as mount
        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrainJobSpec(
            config_path=str(config_file),
            labels_path=str(labels_file),
        )

        errors = validator.validate_train_spec(spec)
        assert errors == []

    def test_path_outside_mounts(self, tmp_path):
        """Test validation fails for path outside allowed mounts."""
        # Create validator with tmp_path as mount
        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrainJobSpec(config_path="/etc/passwd")

        errors = validator.validate_train_spec(spec)
        assert len(errors) == 1
        assert errors[0].field == "config_path"
        assert errors[0].code == "NOT_ALLOWED"
        assert "not within allowed mounts" in errors[0].message

    def test_path_does_not_exist(self, tmp_path):
        """Test validation fails for non-existent path."""
        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        missing_file = tmp_path / "missing.yaml"
        spec = TrainJobSpec(config_path=str(missing_file))

        errors = validator.validate_train_spec(spec)
        assert len(errors) == 1
        assert errors[0].field == "config_path"
        assert errors[0].code == "PATH_NOT_FOUND"
        assert "does not exist" in errors[0].message
        assert errors[0].path == str(missing_file)

    def test_invalid_max_epochs_zero(self, tmp_path):
        """Test validation fails for max_epochs = 0."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("trainer: {}")

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrainJobSpec(config_path=str(config_file), max_epochs=0)

        errors = validator.validate_train_spec(spec)
        assert len(errors) == 1
        assert errors[0].field == "max_epochs"
        assert "between 1 and 10000" in errors[0].message

    def test_invalid_max_epochs_too_high(self, tmp_path):
        """Test validation fails for max_epochs > 10000."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("trainer: {}")

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrainJobSpec(config_path=str(config_file), max_epochs=20000)

        errors = validator.validate_train_spec(spec)
        assert len(errors) == 1
        assert errors[0].field == "max_epochs"

    def test_invalid_batch_size(self, tmp_path):
        """Test validation fails for batch_size = 0."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("trainer: {}")

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrainJobSpec(config_path=str(config_file), batch_size=0)

        errors = validator.validate_train_spec(spec)
        assert len(errors) == 1
        assert errors[0].field == "batch_size"

    def test_invalid_learning_rate_too_high(self, tmp_path):
        """Test validation fails for learning_rate > 1.0."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("trainer: {}")

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrainJobSpec(config_path=str(config_file), learning_rate=5.0)

        errors = validator.validate_train_spec(spec)
        assert len(errors) == 1
        assert errors[0].field == "learning_rate"
        assert "between" in errors[0].message

    def test_multiple_errors(self, tmp_path):
        """Test validation returns all errors."""
        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        missing_config = tmp_path / "missing.yaml"
        missing_labels = tmp_path / "missing.slp"

        spec = TrainJobSpec(
            config_path=str(missing_config),
            labels_path=str(missing_labels),
            max_epochs=0,
            batch_size=0,
        )

        errors = validator.validate_train_spec(spec)
        fields = [e.field for e in errors]

        assert "config_path" in fields
        assert "labels_path" in fields
        assert "max_epochs" in fields
        assert "batch_size" in fields
        assert len(errors) == 4

    def test_valid_resume_checkpoint(self, tmp_path):
        """Test validation passes with valid resume checkpoint."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("trainer: {}")

        checkpoint = tmp_path / "checkpoint.ckpt"
        checkpoint.write_text("checkpoint data")

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrainJobSpec(
            config_path=str(config_file),
            resume_ckpt_path=str(checkpoint),
        )

        errors = validator.validate_train_spec(spec)
        assert errors == []

    def test_missing_resume_checkpoint(self, tmp_path):
        """Test validation fails for missing resume checkpoint."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("trainer: {}")

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        missing_ckpt = tmp_path / "missing.ckpt"
        spec = TrainJobSpec(
            config_path=str(config_file),
            resume_ckpt_path=str(missing_ckpt),
        )

        errors = validator.validate_train_spec(spec)
        assert len(errors) == 1
        assert errors[0].field == "resume_ckpt_path"


class TestJobValidatorConfigContent:
    """Tests for validating TrainJobSpec with config_content."""

    def test_valid_spec_with_config_content(self, tmp_path):
        """Test validation passes with config_content and valid labels_path."""
        labels_file = tmp_path / "labels.slp"
        labels_file.write_text("slp data")

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrainJobSpec(
            config_content="model_config:\n  backbone: unet\n",
            labels_path=str(labels_file),
        )

        errors = validator.validate_train_spec(spec)
        assert errors == []

    def test_valid_spec_with_config_content_no_labels(self, tmp_path):
        """Test validation passes with config_content and no labels_path."""
        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrainJobSpec(config_content="model_config:\n  backbone: unet\n")

        errors = validator.validate_train_spec(spec)
        assert errors == []

    def test_missing_config_error(self, tmp_path):
        """Test validation fails with neither config_paths nor config_content."""
        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        # Create a spec that bypasses __post_init__ validation
        spec = TrainJobSpec.__new__(TrainJobSpec)
        spec.config_path = None
        spec.config_paths = []
        spec.config_content = None
        spec.labels_path = None
        spec.val_labels_path = None
        spec.max_epochs = None
        spec.batch_size = None
        spec.learning_rate = None
        spec.run_name = None
        spec.resume_ckpt_path = None
        spec.path_mappings = {}

        errors = validator.validate_train_spec(spec)
        config_errors = [e for e in errors if e.field == "config"]
        assert len(config_errors) == 1
        assert config_errors[0].code == "MISSING_CONFIG"

    def test_config_content_skips_path_validation(self, tmp_path):
        """Test that config_content specs don't validate config file paths."""
        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        # config_content provided, no config_paths - should not get path errors
        spec = TrainJobSpec(config_content="model_config:\n  backbone: unet\n")

        errors = validator.validate_train_spec(spec)
        config_path_errors = [
            e for e in errors if "config_path" in e.field
        ]
        assert config_path_errors == []

    def test_config_content_with_invalid_labels_path(self, tmp_path):
        """Test validation catches invalid labels_path even with config_content."""
        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        missing_labels = tmp_path / "missing.slp"
        spec = TrainJobSpec(
            config_content="model_config:\n  backbone: unet\n",
            labels_path=str(missing_labels),
        )

        errors = validator.validate_train_spec(spec)
        assert len(errors) == 1
        assert errors[0].field == "labels_path"

    def test_config_content_with_numeric_validation(self, tmp_path):
        """Test numeric validation still works with config_content specs."""
        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrainJobSpec(
            config_content="model_config:\n  backbone: unet\n",
            max_epochs=-1,
            batch_size=0,
        )

        errors = validator.validate_train_spec(spec)
        fields = [e.field for e in errors]
        assert "max_epochs" in fields
        assert "batch_size" in fields


class TestJobValidatorTrackSpec:
    """Tests for validating TrackJobSpec."""

    def test_valid_spec(self, tmp_path):
        """Test validation passes with valid spec."""
        # Create test files
        data_file = tmp_path / "data.slp"
        data_file.write_text("slp data")

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrackJobSpec(
            data_path=str(data_file),
            model_paths=[str(model_dir)],
        )

        errors = validator.validate_track_spec(spec)
        assert errors == []

    def test_missing_data_path(self, tmp_path):
        """Test validation fails for missing data path."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        missing_data = tmp_path / "missing.slp"
        spec = TrackJobSpec(
            data_path=str(missing_data),
            model_paths=[str(model_dir)],
        )

        errors = validator.validate_track_spec(spec)
        assert len(errors) == 1
        assert errors[0].field == "data_path"

    def test_empty_model_paths(self, tmp_path):
        """Test validation fails for empty model_paths."""
        data_file = tmp_path / "data.slp"
        data_file.write_text("slp data")

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrackJobSpec(
            data_path=str(data_file),
            model_paths=[],
        )

        errors = validator.validate_track_spec(spec)
        assert len(errors) == 1
        assert errors[0].field == "model_paths"
        assert "At least one model path" in errors[0].message

    def test_missing_model_path(self, tmp_path):
        """Test validation fails for missing model path."""
        data_file = tmp_path / "data.slp"
        data_file.write_text("slp data")

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        missing_model = tmp_path / "missing_model"
        spec = TrackJobSpec(
            data_path=str(data_file),
            model_paths=[str(missing_model)],
        )

        errors = validator.validate_track_spec(spec)
        assert len(errors) == 1
        assert errors[0].field == "model_paths[0]"

    def test_multiple_model_paths_one_missing(self, tmp_path):
        """Test validation identifies specific missing model path."""
        data_file = tmp_path / "data.slp"
        data_file.write_text("slp data")

        model1 = tmp_path / "model1"
        model1.mkdir()
        missing_model = tmp_path / "model2_missing"

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrackJobSpec(
            data_path=str(data_file),
            model_paths=[str(model1), str(missing_model)],
        )

        errors = validator.validate_track_spec(spec)
        assert len(errors) == 1
        assert errors[0].field == "model_paths[1]"

    def test_invalid_peak_threshold_negative(self, tmp_path):
        """Test validation fails for negative peak_threshold."""
        data_file = tmp_path / "data.slp"
        data_file.write_text("slp data")

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrackJobSpec(
            data_path=str(data_file),
            model_paths=[str(model_dir)],
            peak_threshold=-0.1,
        )

        errors = validator.validate_track_spec(spec)
        assert len(errors) == 1
        assert errors[0].field == "peak_threshold"

    def test_invalid_peak_threshold_over_one(self, tmp_path):
        """Test validation fails for peak_threshold > 1.0."""
        data_file = tmp_path / "data.slp"
        data_file.write_text("slp data")

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrackJobSpec(
            data_path=str(data_file),
            model_paths=[str(model_dir)],
            peak_threshold=1.5,
        )

        errors = validator.validate_track_spec(spec)
        assert len(errors) == 1
        assert errors[0].field == "peak_threshold"

    def test_valid_output_path(self, tmp_path):
        """Test validation passes when output directory exists."""
        data_file = tmp_path / "data.slp"
        data_file.write_text("slp data")

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrackJobSpec(
            data_path=str(data_file),
            model_paths=[str(model_dir)],
            output_path=str(output_dir / "predictions.slp"),
        )

        errors = validator.validate_track_spec(spec)
        assert errors == []

    def test_output_path_parent_missing(self, tmp_path):
        """Test validation fails when output directory doesn't exist."""
        data_file = tmp_path / "data.slp"
        data_file.write_text("slp data")

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrackJobSpec(
            data_path=str(data_file),
            model_paths=[str(model_dir)],
            output_path=str(tmp_path / "missing_dir" / "predictions.slp"),
        )

        errors = validator.validate_track_spec(spec)
        assert len(errors) == 1
        assert "output_path" in errors[0].field


class TestJobValidatorGeneric:
    """Tests for generic validate() method."""

    def test_validate_train_spec(self, tmp_path):
        """Test generic validate() with TrainJobSpec."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("trainer: {}")

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrainJobSpec(config_path=str(config_file))
        errors = validator.validate(spec)
        assert errors == []

    def test_validate_track_spec(self, tmp_path):
        """Test generic validate() with TrackJobSpec."""
        data_file = tmp_path / "data.slp"
        data_file.write_text("slp data")

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        spec = TrackJobSpec(
            data_path=str(data_file),
            model_paths=[str(model_dir)],
        )
        errors = validator.validate(spec)
        assert errors == []

    def test_validate_unknown_type(self, tmp_path):
        """Test generic validate() with unknown type."""
        mounts = [MockMount(path=str(tmp_path), label="test")]
        validator = JobValidator(mounts=mounts)

        # Pass a non-spec object
        errors = validator.validate("not a spec")
        assert len(errors) == 1
        assert errors[0].field == "type"
        assert "Unknown job spec type" in errors[0].message


class TestNumericConstraints:
    """Tests for numeric constraint definitions."""

    def test_max_epochs_constraints(self):
        """Test max_epochs constraint values."""
        assert NUMERIC_CONSTRAINTS["max_epochs"]["min"] == 1
        assert NUMERIC_CONSTRAINTS["max_epochs"]["max"] == 10000

    def test_batch_size_constraints(self):
        """Test batch_size constraint values."""
        assert NUMERIC_CONSTRAINTS["batch_size"]["min"] == 1
        assert NUMERIC_CONSTRAINTS["batch_size"]["max"] == 256

    def test_learning_rate_constraints(self):
        """Test learning_rate constraint values."""
        assert NUMERIC_CONSTRAINTS["learning_rate"]["min"] == 1e-10
        assert NUMERIC_CONSTRAINTS["learning_rate"]["max"] == 1.0

    def test_peak_threshold_constraints(self):
        """Test peak_threshold constraint values."""
        assert NUMERIC_CONSTRAINTS["peak_threshold"]["min"] == 0.0
        assert NUMERIC_CONSTRAINTS["peak_threshold"]["max"] == 1.0
