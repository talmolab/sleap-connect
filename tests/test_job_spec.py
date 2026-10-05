"""Tests for job specification data types."""

import json
import pytest

from sleap_rtc.jobs.spec import TrainJobSpec, TrackJobSpec, parse_job_spec


class TestTrainJobSpec:
    """Tests for TrainJobSpec dataclass."""

    def test_create_with_required_fields(self):
        """Test creating spec with only required config_path."""
        spec = TrainJobSpec(config_path="/vast/project/centroid.yaml")

        assert spec.config_path == "/vast/project/centroid.yaml"
        assert spec.labels_path is None
        assert spec.val_labels_path is None
        assert spec.max_epochs is None
        assert spec.batch_size is None
        assert spec.learning_rate is None
        assert spec.run_name is None
        assert spec.resume_ckpt_path is None

    def test_create_with_all_fields(self):
        """Test creating spec with all fields populated."""
        spec = TrainJobSpec(
            config_path="/vast/project/centroid.yaml",
            labels_path="/vast/data/labels.slp",
            val_labels_path="/vast/data/val_labels.slp",
            max_epochs=100,
            batch_size=8,
            learning_rate=0.0001,
            run_name="experiment-1",
            resume_ckpt_path="/vast/models/checkpoint.ckpt",
        )

        assert spec.config_path == "/vast/project/centroid.yaml"
        assert spec.labels_path == "/vast/data/labels.slp"
        assert spec.val_labels_path == "/vast/data/val_labels.slp"
        assert spec.max_epochs == 100
        assert spec.batch_size == 8
        assert spec.learning_rate == 0.0001
        assert spec.run_name == "experiment-1"
        assert spec.resume_ckpt_path == "/vast/models/checkpoint.ckpt"

    def test_to_json_minimal(self):
        """Test JSON serialization with minimal fields."""
        spec = TrainJobSpec(config_path="/vast/project/centroid.yaml")
        json_str = spec.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "train"
        assert parsed["config_path"] == "/vast/project/centroid.yaml"
        # None values should be omitted
        assert "labels_path" not in parsed
        assert "max_epochs" not in parsed

    def test_to_json_full(self):
        """Test JSON serialization with all fields."""
        spec = TrainJobSpec(
            config_path="/vast/project/centroid.yaml",
            labels_path="/vast/data/labels.slp",
            max_epochs=100,
            batch_size=8,
            learning_rate=0.0001,
            run_name="exp1",
        )
        json_str = spec.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "train"
        assert parsed["config_path"] == "/vast/project/centroid.yaml"
        assert parsed["labels_path"] == "/vast/data/labels.slp"
        assert parsed["max_epochs"] == 100
        assert parsed["batch_size"] == 8
        assert parsed["learning_rate"] == 0.0001
        assert parsed["run_name"] == "exp1"

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = json.dumps(
            {
                "type": "train",
                "config_path": "/vast/project/centroid.yaml",
                "labels_path": "/vast/data/labels.slp",
                "max_epochs": 100,
            }
        )
        spec = TrainJobSpec.from_json(json_str)

        assert spec.config_path == "/vast/project/centroid.yaml"
        assert spec.labels_path == "/vast/data/labels.slp"
        assert spec.max_epochs == 100
        assert spec.batch_size is None

    def test_roundtrip(self):
        """Test JSON serialization roundtrip."""
        original = TrainJobSpec(
            config_path="/vast/project/centroid.yaml",
            labels_path="/vast/data/labels.slp",
            max_epochs=100,
            batch_size=8,
        )
        json_str = original.to_json()
        restored = TrainJobSpec.from_json(json_str)

        assert restored.config_path == original.config_path
        assert restored.labels_path == original.labels_path
        assert restored.max_epochs == original.max_epochs
        assert restored.batch_size == original.batch_size

    def test_to_dict(self):
        """Test dictionary conversion."""
        spec = TrainJobSpec(config_path="/vast/config.yaml", max_epochs=50)
        d = spec.to_dict()

        assert d["type"] == "train"
        assert d["config_path"] == "/vast/config.yaml"
        assert d["max_epochs"] == 50
        assert "labels_path" not in d  # None values omitted

    def test_from_dict(self):
        """Test creating spec from dictionary."""
        d = {
            "type": "train",
            "config_path": "/vast/config.yaml",
            "max_epochs": 50,
        }
        spec = TrainJobSpec.from_dict(d)

        assert spec.config_path == "/vast/config.yaml"
        assert spec.max_epochs == 50


class TestTrainJobSpecConfigContent:
    """Tests for TrainJobSpec config_content and path_mappings fields."""

    def test_create_with_config_content(self):
        """Test creating spec with config_content instead of config_path."""
        spec = TrainJobSpec(
            config_content="model_config:\n  backbone: unet\n",
            labels_path="/mnt/data/labels.slp",
        )

        assert spec.config_content == "model_config:\n  backbone: unet\n"
        assert spec.labels_path == "/mnt/data/labels.slp"
        assert spec.config_paths == []
        assert spec.config_path is None

    def test_create_with_no_config_raises(self):
        """Test that creating spec with no config source raises ValueError."""
        with pytest.raises(ValueError, match="config_path.*config_paths.*config_content"):
            TrainJobSpec(labels_path="/mnt/data/labels.slp")

    def test_create_with_path_mappings(self):
        """Test creating spec with path_mappings."""
        mappings = {
            "C:/Users/data/video.mp4": "/mnt/shared/video.mp4",
            "C:/Users/data/video2.mp4": "/mnt/shared/video2.mp4",
        }
        spec = TrainJobSpec(
            config_content="model_config:\n  backbone: unet\n",
            labels_path="/mnt/data/labels.slp",
            path_mappings=mappings,
        )

        assert spec.path_mappings == mappings

    def test_default_path_mappings_is_empty(self):
        """Test that path_mappings defaults to empty dict."""
        spec = TrainJobSpec(config_path="/vast/config.yaml")
        assert spec.path_mappings == {}

    def test_to_json_with_config_content(self):
        """Test JSON serialization includes config_content."""
        spec = TrainJobSpec(
            config_content="model_config:\n  backbone: unet\n",
            labels_path="/mnt/data/labels.slp",
        )
        json_str = spec.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "train"
        assert parsed["config_content"] == "model_config:\n  backbone: unet\n"
        assert parsed["labels_path"] == "/mnt/data/labels.slp"
        assert "config_paths" not in parsed  # empty list omitted
        assert "config_path" not in parsed  # None omitted

    def test_to_json_with_path_mappings(self):
        """Test JSON serialization includes path_mappings."""
        mappings = {"C:/data/vid.mp4": "/mnt/data/vid.mp4"}
        spec = TrainJobSpec(
            config_content="backbone: unet\n",
            path_mappings=mappings,
        )
        json_str = spec.to_json()
        parsed = json.loads(json_str)

        assert parsed["path_mappings"] == mappings

    def test_to_json_omits_empty_path_mappings(self):
        """Test JSON serialization omits empty path_mappings."""
        spec = TrainJobSpec(config_path="/vast/config.yaml")
        json_str = spec.to_json()
        parsed = json.loads(json_str)

        assert "path_mappings" not in parsed

    def test_from_json_with_config_content(self):
        """Test JSON deserialization with config_content."""
        json_str = json.dumps({
            "type": "train",
            "config_content": "backbone: unet\n",
            "labels_path": "/mnt/data/labels.slp",
        })
        spec = TrainJobSpec.from_json(json_str)

        assert spec.config_content == "backbone: unet\n"
        assert spec.labels_path == "/mnt/data/labels.slp"
        assert spec.config_paths == []

    def test_from_json_with_path_mappings(self):
        """Test JSON deserialization with path_mappings."""
        mappings = {"C:/data/vid.mp4": "/mnt/data/vid.mp4"}
        json_str = json.dumps({
            "type": "train",
            "config_content": "backbone: unet\n",
            "path_mappings": mappings,
        })
        spec = TrainJobSpec.from_json(json_str)

        assert spec.path_mappings == mappings

    def test_roundtrip_config_content(self):
        """Test JSON roundtrip with config_content and path_mappings."""
        mappings = {"C:/data/vid.mp4": "/mnt/data/vid.mp4"}
        original = TrainJobSpec(
            config_content="model_config:\n  backbone: unet\n",
            labels_path="/mnt/data/labels.slp",
            path_mappings=mappings,
            max_epochs=100,
        )
        json_str = original.to_json()
        restored = TrainJobSpec.from_json(json_str)

        assert restored.config_content == original.config_content
        assert restored.labels_path == original.labels_path
        assert restored.path_mappings == original.path_mappings
        assert restored.max_epochs == original.max_epochs
        assert restored.config_paths == []

    def test_to_dict_with_config_content(self):
        """Test dict conversion with config_content."""
        spec = TrainJobSpec(
            config_content="backbone: unet\n",
            labels_path="/mnt/data/labels.slp",
        )
        d = spec.to_dict()

        assert d["type"] == "train"
        assert d["config_content"] == "backbone: unet\n"
        assert "config_paths" not in d
        assert "path_mappings" not in d  # empty dict omitted

    def test_from_dict_with_config_content(self):
        """Test creating spec from dict with config_content."""
        d = {
            "type": "train",
            "config_content": "backbone: unet\n",
            "path_mappings": {"a": "b"},
        }
        spec = TrainJobSpec.from_dict(d)

        assert spec.config_content == "backbone: unet\n"
        assert spec.path_mappings == {"a": "b"}

    def test_parse_job_spec_with_config_content(self):
        """Test parse_job_spec handles config_content."""
        json_str = json.dumps({
            "type": "train",
            "config_content": "backbone: unet\n",
            "labels_path": "/mnt/data/labels.slp",
        })
        spec = parse_job_spec(json_str)

        assert isinstance(spec, TrainJobSpec)
        assert spec.config_content == "backbone: unet\n"

    def test_from_json_ignores_unknown_fields(self):
        """Test that from_json ignores unknown fields for forward compatibility."""
        json_str = json.dumps({
            "type": "train",
            "config_path": "/vast/config.yaml",
            "unknown_future_field": "some_value",
        })
        spec = TrainJobSpec.from_json(json_str)

        assert spec.config_path == "/vast/config.yaml"
        assert not hasattr(spec, "unknown_future_field")


class TestTrackJobSpec:
    """Tests for TrackJobSpec dataclass."""

    def test_create_with_required_fields(self):
        """Test creating spec with required fields."""
        spec = TrackJobSpec(
            data_path="/vast/data/labels.slp",
            model_paths=["/vast/models/centroid"],
        )

        assert spec.data_path == "/vast/data/labels.slp"
        assert spec.model_paths == ["/vast/models/centroid"]
        assert spec.output_path is None
        assert spec.batch_size is None
        assert spec.peak_threshold is None
        assert spec.only_suggested_frames is False
        assert spec.frames is None

    def test_create_with_all_fields(self):
        """Test creating spec with all fields."""
        spec = TrackJobSpec(
            data_path="/vast/data/labels.slp",
            model_paths=["/vast/models/centroid", "/vast/models/instance"],
            output_path="/vast/predictions.slp",
            batch_size=8,
            peak_threshold=0.3,
            only_suggested_frames=True,
            frames="0-100,200-300",
        )

        assert spec.data_path == "/vast/data/labels.slp"
        assert len(spec.model_paths) == 2
        assert spec.output_path == "/vast/predictions.slp"
        assert spec.batch_size == 8
        assert spec.peak_threshold == 0.3
        assert spec.only_suggested_frames is True
        assert spec.frames == "0-100,200-300"

    def test_to_json_minimal(self):
        """Test JSON serialization with minimal fields."""
        spec = TrackJobSpec(
            data_path="/vast/data/labels.slp",
            model_paths=["/vast/models/centroid"],
        )
        json_str = spec.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "track"
        assert parsed["data_path"] == "/vast/data/labels.slp"
        assert parsed["model_paths"] == ["/vast/models/centroid"]
        # False/None values should be omitted
        assert "only_suggested_frames" not in parsed
        assert "output_path" not in parsed

    def test_to_json_full(self):
        """Test JSON serialization with all fields."""
        spec = TrackJobSpec(
            data_path="/vast/data/labels.slp",
            model_paths=["/vast/models/centroid", "/vast/models/instance"],
            output_path="/vast/predictions.slp",
            batch_size=8,
            peak_threshold=0.3,
            only_suggested_frames=True,
            frames="0-100",
        )
        json_str = spec.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "track"
        assert parsed["data_path"] == "/vast/data/labels.slp"
        assert parsed["model_paths"] == [
            "/vast/models/centroid",
            "/vast/models/instance",
        ]
        assert parsed["output_path"] == "/vast/predictions.slp"
        assert parsed["batch_size"] == 8
        assert parsed["peak_threshold"] == 0.3
        assert parsed["only_suggested_frames"] is True
        assert parsed["frames"] == "0-100"

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = json.dumps(
            {
                "type": "track",
                "data_path": "/vast/data/labels.slp",
                "model_paths": ["/vast/models/centroid"],
                "batch_size": 16,
            }
        )
        spec = TrackJobSpec.from_json(json_str)

        assert spec.data_path == "/vast/data/labels.slp"
        assert spec.model_paths == ["/vast/models/centroid"]
        assert spec.batch_size == 16
        assert spec.only_suggested_frames is False

    def test_roundtrip(self):
        """Test JSON serialization roundtrip."""
        original = TrackJobSpec(
            data_path="/vast/data/labels.slp",
            model_paths=["/vast/models/centroid", "/vast/models/instance"],
            batch_size=8,
            only_suggested_frames=True,
        )
        json_str = original.to_json()
        restored = TrackJobSpec.from_json(json_str)

        assert restored.data_path == original.data_path
        assert restored.model_paths == original.model_paths
        assert restored.batch_size == original.batch_size
        assert restored.only_suggested_frames == original.only_suggested_frames

    def test_multiple_model_paths(self):
        """Test spec with multiple model paths."""
        spec = TrackJobSpec(
            data_path="/vast/data.slp",
            model_paths=[
                "/vast/models/centroid",
                "/vast/models/centered_instance",
                "/vast/models/topdown",
            ],
        )
        json_str = spec.to_json()
        parsed = json.loads(json_str)

        assert len(parsed["model_paths"]) == 3
        assert "/vast/models/centroid" in parsed["model_paths"]
        assert "/vast/models/centered_instance" in parsed["model_paths"]
        assert "/vast/models/topdown" in parsed["model_paths"]

    def test_to_dict(self):
        """Test dictionary conversion."""
        spec = TrackJobSpec(
            data_path="/vast/data.slp",
            model_paths=["/vast/models/model"],
            batch_size=16,
        )
        d = spec.to_dict()

        assert d["type"] == "track"
        assert d["data_path"] == "/vast/data.slp"
        assert d["model_paths"] == ["/vast/models/model"]
        assert d["batch_size"] == 16
        assert "only_suggested_frames" not in d  # False value omitted

    def test_from_dict(self):
        """Test creating spec from dictionary."""
        d = {
            "type": "track",
            "data_path": "/vast/data.slp",
            "model_paths": ["/vast/models/model"],
        }
        spec = TrackJobSpec.from_dict(d)

        assert spec.data_path == "/vast/data.slp"
        assert spec.model_paths == ["/vast/models/model"]


class TestTrackJobSpecFrameFilter:
    """Tests for the new frame_filter and video_index fields."""

    def _base_spec_kwargs(self):
        return dict(
            data_path="/path/to/data.slp",
            model_paths=["/path/to/model"],
            output_path="/path/to/out.predictions.slp",
        )

    def test_frame_filter_defaults_to_none(self):
        spec = TrackJobSpec(**self._base_spec_kwargs())
        assert spec.frame_filter is None

    def test_video_index_defaults_to_none(self):
        spec = TrackJobSpec(**self._base_spec_kwargs())
        assert spec.video_index is None

    @pytest.mark.parametrize(
        "value", ["suggested", "user", "predicted", "random", None]
    )
    def test_frame_filter_accepts_valid_values(self, value):
        spec = TrackJobSpec(**self._base_spec_kwargs(), frame_filter=value)
        assert spec.frame_filter == value

    def test_frame_filter_rejects_invalid_value(self):
        with pytest.raises(ValueError, match="frame_filter"):
            TrackJobSpec(**self._base_spec_kwargs(), frame_filter="not-a-real-filter")

    def test_only_suggested_frames_true_migrates_to_frame_filter(self):
        """Backward compat: only_suggested_frames=True (deprecated) maps to frame_filter='suggested'."""
        spec = TrackJobSpec(**self._base_spec_kwargs(), only_suggested_frames=True)
        assert spec.frame_filter == "suggested"

    def test_only_suggested_frames_does_not_overwrite_explicit_frame_filter(self):
        """If both set, explicit frame_filter wins (forward-compatible)."""
        spec = TrackJobSpec(
            **self._base_spec_kwargs(),
            only_suggested_frames=True,
            frame_filter="user",
        )
        assert spec.frame_filter == "user"

    def test_video_index_accepts_int(self):
        spec = TrackJobSpec(**self._base_spec_kwargs(), video_index=2)
        assert spec.video_index == 2


class TestParseJobSpec:
    """Tests for parse_job_spec utility function."""

    def test_parse_train_spec(self):
        """Test parsing a train job spec."""
        json_str = json.dumps(
            {
                "type": "train",
                "config_path": "/vast/config.yaml",
                "max_epochs": 100,
            }
        )
        spec = parse_job_spec(json_str)

        assert isinstance(spec, TrainJobSpec)
        assert spec.config_path == "/vast/config.yaml"
        assert spec.max_epochs == 100

    def test_parse_track_spec(self):
        """Test parsing a track job spec."""
        json_str = json.dumps(
            {
                "type": "track",
                "data_path": "/vast/data.slp",
                "model_paths": ["/vast/models/model"],
            }
        )
        spec = parse_job_spec(json_str)

        assert isinstance(spec, TrackJobSpec)
        assert spec.data_path == "/vast/data.slp"
        assert spec.model_paths == ["/vast/models/model"]

    def test_parse_unknown_type(self):
        """Test parsing with unknown type raises error."""
        json_str = json.dumps({"type": "unknown", "some_field": "value"})

        with pytest.raises(ValueError, match="Unknown job type"):
            parse_job_spec(json_str)

    def test_parse_missing_type(self):
        """Test parsing without type raises error."""
        json_str = json.dumps({"config_path": "/vast/config.yaml"})

        with pytest.raises(ValueError, match="Unknown job type"):
            parse_job_spec(json_str)


class TestProjectTag:
    """Specs carry the submitting project's identity, for clients' job lists."""

    @pytest.mark.parametrize(
        "base",
        [
            {"type": "train", "config_path": "/c.yaml"},
            {"type": "track", "data_path": "/d.slp", "model_paths": ["/m"]},
        ],
    )
    def test_round_trips(self, base):
        project = {"name": "flies.slp", "id": "abc123"}
        spec = parse_job_spec(json.dumps({**base, "project": project}))
        assert spec.project == project
        assert parse_job_spec(spec.to_json()).project == project

    def test_absent_is_none_and_not_serialized(self):
        spec = parse_job_spec(json.dumps({"type": "train", "config_path": "/c.yaml"}))
        assert spec.project is None
        assert "project" not in spec.to_dict()


class TestRunTag:
    """Specs carry a run id linking jobs submitted together (e.g. top-down)."""

    @pytest.mark.parametrize(
        "base",
        [
            {"type": "train", "config_path": "/c.yaml"},
            {"type": "track", "data_path": "/d.slp", "model_paths": ["/m"]},
        ],
    )
    def test_round_trips(self, base):
        run = {"id": "run-abc123", "index": 0, "count": 2}
        spec = parse_job_spec(json.dumps({**base, "run": run}))
        assert spec.run == run
        assert parse_job_spec(spec.to_json()).run == run

    def test_absent_is_none_and_not_serialized(self):
        spec = parse_job_spec(json.dumps({"type": "train", "config_path": "/c.yaml"}))
        assert spec.run is None


class TestPostInference:
    """TrainJobSpec carries inference jobs to chain once its run completes."""

    def test_round_trips(self):
        post_inference = [{"frame_filter": "suggested", "tracker": "simple"}]
        spec = parse_job_spec(
            json.dumps(
                {
                    "type": "train",
                    "config_path": "/c.yaml",
                    "post_inference": post_inference,
                }
            )
        )
        assert spec.post_inference == post_inference
        assert parse_job_spec(spec.to_json()).post_inference == post_inference

    def test_absent_is_none_and_not_serialized(self):
        spec = parse_job_spec(json.dumps({"type": "train", "config_path": "/c.yaml"}))
        assert spec.post_inference is None
        assert "post_inference" not in spec.to_dict()

    def test_not_a_field_on_track_job_spec(self):
        """Only a train job's completion can trigger chaining."""
        spec = parse_job_spec(
            json.dumps({"type": "track", "data_path": "/d.slp", "model_paths": ["/m"]})
        )
        assert not hasattr(spec, "post_inference")
        assert "run" not in spec.to_dict()
