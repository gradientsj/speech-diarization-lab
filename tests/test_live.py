"""Live chunking and online speaker tracking, with stubbed models."""

import numpy as np

from diarlab.diarize import ClusteredConfig
from diarlab.live import SAMPLE_RATE, LiveSession, OnlineSpeakerTracker
from diarlab.types import Region, Word


def _unit(dim, axis):
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


class TestTracker:
    def test_first_cluster_gets_id_zero(self):
        tr = OnlineSpeakerTracker(0.6)
        emb = np.stack([_unit(8, 0), _unit(8, 0)])
        assert tr.assign(emb, [0, 0]) == [0, 0]

    def test_same_voice_keeps_its_id_across_chunks(self):
        tr = OnlineSpeakerTracker(0.6)
        tr.assign(np.stack([_unit(8, 0)]), [0])
        tr.assign(np.stack([_unit(8, 1)]), [0])  # second speaker -> id 1
        # speaker 0 again in a later chunk, slightly perturbed
        noisy = _unit(8, 0) + 0.05
        noisy /= np.linalg.norm(noisy)
        assert tr.assign(np.stack([noisy]), [0]) == [0]

    def test_distinct_voice_mints_new_id(self):
        tr = OnlineSpeakerTracker(0.6)
        tr.assign(np.stack([_unit(8, 0)]), [0])
        assert tr.assign(np.stack([_unit(8, 1)]), [0]) == [1]

    def test_two_local_clusters_map_independently(self):
        tr = OnlineSpeakerTracker(0.6)
        tr.assign(np.stack([_unit(8, 0)]), [0])
        emb = np.stack([_unit(8, 1), _unit(8, 0)])
        # local labels are chunk-relative; global ids must not collide
        assert tr.assign(emb, [0, 1]) == [1, 0]

    def test_centroid_updates_are_weighted(self):
        tr = OnlineSpeakerTracker(0.6)
        tr.assign(np.stack([_unit(8, 0)] * 9), [0] * 9)
        tilted = (_unit(8, 0) + _unit(8, 1)) / np.linalg.norm(_unit(8, 0) + _unit(8, 1))
        tr.assign(np.stack([tilted]), [0])
        # nine on-axis windows outweigh one tilted window
        assert tr.centroids[0][0] > 0.9


def _session(speaker_axis_by_chunk):
    """A LiveSession whose models pretend each chunk is one steady speaker."""
    state = {"chunk": -1}

    def vad(chunk):
        state["chunk"] += 1  # vad runs first in every chunk
        return [Region(0.5, len(chunk) / SAMPLE_RATE - 0.5)]

    def embed(chunk, windows):
        axis = speaker_axis_by_chunk[state["chunk"]]
        return np.stack([_unit(8, axis)] * len(windows))

    def transcribe(chunk):
        return [Word(1.0, 1.5, f"chunk{state['chunk']}")]

    return LiveSession(
        vad, embed, transcribe,
        config=ClusteredConfig(distance_threshold=0.6, vad_pad=0.05),
        chunk_seconds=5.0,
    )


def test_feed_processes_only_complete_chunks():
    session = _session([0])
    assert session.feed(np.zeros(3 * SAMPLE_RATE, dtype=np.float32)) == []
    segments = session.feed(np.zeros(2 * SAMPLE_RATE, dtype=np.float32))
    assert len(segments) == 1
    assert segments[0].text == "chunk0"


def test_times_are_absolute_across_chunks():
    session = _session([0, 0])
    session.feed(np.zeros(5 * SAMPLE_RATE, dtype=np.float32))
    segments = session.feed(np.zeros(5 * SAMPLE_RATE, dtype=np.float32))
    # second chunk's word at +1.0s lands at 6.0s absolute
    assert segments[0].words[0].start == 6.0


def test_speaker_identity_stable_across_chunks():
    session = _session([0, 1, 0])
    first = session.feed(np.zeros(5 * SAMPLE_RATE, dtype=np.float32))
    second = session.feed(np.zeros(5 * SAMPLE_RATE, dtype=np.float32))
    third = session.feed(np.zeros(5 * SAMPLE_RATE, dtype=np.float32))
    assert first[0].speaker == "SPEAKER_00"
    assert second[0].speaker == "SPEAKER_01"
    assert third[0].speaker == "SPEAKER_00"  # the voice came back


def test_flush_handles_the_tail():
    session = _session([0])
    session.feed(np.zeros(2 * SAMPLE_RATE, dtype=np.float32))
    segments = session.flush()
    assert len(segments) == 1
    assert session.flush() == []  # idempotent on empty buffer
