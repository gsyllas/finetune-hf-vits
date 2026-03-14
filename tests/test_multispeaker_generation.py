"""
Tests for multi-speaker full-generation logic in run_vits_finetuning.py.
Covers the bugs we hit in production:
  1. Passing a list of speaker_ids to a batch-1 model call → ValueError
  2. np.concatenate on variable-length waveforms → shape mismatch
"""
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers that replicate the logic from run_vits_finetuning.py
# ---------------------------------------------------------------------------

def build_full_generation_waveform(model, full_generation_sample, num_speakers, device="cpu"):
    """Exact copy of the logic used in run_vits_finetuning.py."""
    import torch
    with torch.no_grad():
        if num_speakers < 2:
            full_generation_waveform = (
                model(**full_generation_sample, speaker_id=None)
                .waveform.cpu()
                .numpy()
            )
        else:
            full_generation_waveform = [
                model(**full_generation_sample, speaker_id=sid)
                .waveform.cpu()
                .numpy()[0]
                for sid in range(min(5, num_speakers))
            ]
    return full_generation_waveform


def make_mock_model(waveform_lengths):
    """
    Returns a mock model whose waveform output length cycles through
    waveform_lengths (simulating different speakers producing different
    durations).
    """
    import torch
    call_count = [0]

    def mock_forward(**kwargs):
        length = waveform_lengths[call_count[0] % len(waveform_lengths)]
        call_count[0] += 1
        out = SimpleNamespace()
        out.waveform = torch.zeros(1, 1, length)
        return out

    model = MagicMock(side_effect=mock_forward)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSingleSpeakerGeneration:
    def test_returns_numpy_array(self):
        model = make_mock_model([22050])
        result = build_full_generation_waveform(model, {}, num_speakers=1)
        assert isinstance(result, np.ndarray)

    def test_shape_is_3d(self):
        """Single-speaker: shape [1, 1, T] — iterable as [1, T] per element."""
        model = make_mock_model([22050])
        result = build_full_generation_waveform(model, {}, num_speakers=1)
        assert result.ndim == 3
        assert result.shape == (1, 1, 22050)

    def test_called_once_with_none_speaker_id(self):
        model = make_mock_model([22050])
        build_full_generation_waveform(model, {}, num_speakers=1)
        assert model.call_count == 1
        assert model.call_args.kwargs["speaker_id"] is None


class TestMultiSpeakerGeneration:
    def test_returns_list(self):
        model = make_mock_model([22050, 21000])
        result = build_full_generation_waveform(model, {}, num_speakers=2)
        assert isinstance(result, list)

    def test_list_length_equals_num_speakers(self):
        model = make_mock_model([22050, 21000, 19000])
        result = build_full_generation_waveform(model, {}, num_speakers=3)
        assert len(result) == 3

    def test_capped_at_five_speakers(self):
        lengths = [22050] * 10
        model = make_mock_model(lengths)
        result = build_full_generation_waveform(model, {}, num_speakers=10)
        assert len(result) == 5
        assert model.call_count == 5

    def test_variable_length_waveforms_do_not_crash(self):
        """Regression: np.concatenate would fail here; list handles it fine."""
        lengths = [36352, 35840]  # the exact sizes that crashed in prod
        model = make_mock_model(lengths)
        result = build_full_generation_waveform(model, {}, num_speakers=2)
        assert len(result) == 2
        assert result[0].shape[-1] == 36352
        assert result[1].shape[-1] == 35840

    def test_each_element_shape_is_2d(self):
        """Each list element should be [1, T] so log_on_trackers can iterate."""
        model = make_mock_model([22050, 21000])
        result = build_full_generation_waveform(model, {}, num_speakers=2)
        for elem in result:
            assert elem.ndim == 2, f"Expected 2D array, got shape {elem.shape}"

    def test_speaker_ids_are_sequential_integers(self):
        model = make_mock_model([22050] * 3)
        build_full_generation_waveform(model, {}, num_speakers=3)
        called_ids = [call.kwargs["speaker_id"] for call in model.call_args_list]
        assert called_ids == [0, 1, 2]

    def test_never_passes_list_as_speaker_id(self):
        """Regression: passing a list to the model causes a ValueError."""
        model = make_mock_model([22050, 21000])
        build_full_generation_waveform(model, {}, num_speakers=2)
        for call in model.call_args_list:
            sid = call.kwargs["speaker_id"]
            assert not isinstance(sid, list), (
                f"speaker_id must be a scalar, not a list: {sid}"
            )


class TestLogOnTrackersCompatibility:
    """Ensure the waveform shape returned by build_full_generation_waveform
    is iterable in the same way log_on_trackers expects."""

    def _collect_audio_shapes(self, full_generation_waveform):
        shapes = []
        for w in full_generation_waveform:
            shapes.append(np.asarray(w).shape)
        return shapes

    def test_single_speaker_iteration_gives_2d_elements(self):
        model = make_mock_model([22050])
        result = build_full_generation_waveform(model, {}, num_speakers=1)
        shapes = self._collect_audio_shapes(result)
        assert len(shapes) == 1
        assert len(shapes[0]) == 2  # [1, T]

    def test_multi_speaker_iteration_gives_2d_elements(self):
        model = make_mock_model([22050, 21000])
        result = build_full_generation_waveform(model, {}, num_speakers=2)
        shapes = self._collect_audio_shapes(result)
        assert len(shapes) == 2
        for s in shapes:
            assert len(s) == 2  # [1, T]
