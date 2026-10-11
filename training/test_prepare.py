import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np

from opensv_train.prepare import alignment, crop, prepare, prepare_batch
from opensv_train.score import load_score, QUARTER


class PreparationTests(unittest.TestCase):
    def test_missing_dictionary_is_reported_before_reading_audio(self):
        project = dict(tracks=[dict(mainGroup=dict(notes=[dict(lyrics="chen")]))])
        args = Namespace(offset=0, project="song.vsqx", dictionary=None)
        with patch("opensv_train.prepare.load_score", return_value=(project, [])), patch("opensv_train.prepare.read_audio") as read:
            with self.assertRaisesRegex(ValueError, "缺少发音词典"):
                prepare(args)
            read.assert_not_called()

    def test_engine_error_preserves_stderr(self):
        project = dict(tracks=[dict(mainGroup=dict(notes=[dict(phonemes="a")]))])
        result = Namespace(returncode=1, stdout="", stderr='Error: unsupported phoneme "bad"')
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(offset=0, project="song.svp", dictionary=None, audio="song.wav",
                             output=Path(directory) / "dataset", voice="voice", engine="opensv", language="mandarin")
            with patch("opensv_train.prepare.load_score", return_value=(project, [])), \
                 patch("opensv_train.prepare.read_audio", return_value=(16000, np.zeros(16000), [])), \
                 patch("opensv_train.prepare.alignment", return_value=([], [], {})), \
                 patch("opensv_train.prepare.digest", return_value="digest"), \
                 patch("opensv_train.prepare.subprocess.run", return_value=result):
                with self.assertRaisesRegex(RuntimeError, 'unsupported phoneme "bad"'):
                    prepare(args)
                self.assertFalse((args.output / "dataset.json").exists())

    def test_batch_splits_sources_and_rejects_duplicates(self):
        def source(args):
            root = args.output
            root.mkdir()
            (root / "a.wav").write_bytes(args.audio.encode())
            for name in ("base_model.dnni", "pitch_model.dnni"):
                (root / name).write_bytes(b"model")
            (root / "dataset.json").write_text(json.dumps(dict(samples=[dict(id="a", audio="a.wav", features="a.npz")])), encoding="utf-8")
            (root / "preparation.json").write_text(json.dumps(dict(warnings=[], errors=[], audio_sha256=args.audio,
                segments=[dict(id="a", start=0, end=1, split="train")])), encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory, patch("opensv_train.prepare.prepare", side_effect=source), patch("opensv_train.prepare.emit"):
            output = Path(directory) / "good"
            prepare_batch(Namespace(output=output), [dict(audio="first"), dict(audio="second")])
            samples = json.loads((output / "dataset.json").read_text(encoding="utf-8"))["samples"]
            self.assertEqual([s["split"] for s in samples], ["train", "validation"])
            self.assertEqual(samples[1]["audio"], "source-002/a.wav")
            output = Path(directory) / "duplicate"
            prepare_batch(Namespace(output=output), [dict(audio="first"), dict(audio="first")])
            self.assertFalse((output / "dataset.json").exists())
            self.assertTrue(json.loads((output / "preparation.json").read_text(encoding="utf-8"))["errors"])

    def test_vsqx_preroll_and_tempo(self):
        for version in (3, 4):
            names = ("t", "v", "m", "nu", "de", "vsPart", "dur", "n", "y") if version == 4 else (
                "posTick", "bpm", "posMes", "nume", "denomi", "musicalPart", "durTick", "noteNum", "lyric")
            tick, bpm, measure, num, den, part, dur, pitch, lyric = names
            xml = f'''<vsq{version} xmlns="urn:test"><masterTrack><resolution>480</resolution><preMeasure>1</preMeasure>
            <timeSig><{measure}>0</{measure}><{num}>4</{num}><{den}>4</{den}></timeSig>
            <tempo><{tick}>0</{tick}><{bpm}>12000</{bpm}></tempo>
            <tempo><{tick}>1920</{tick}><{bpm}>6000</{bpm}></tempo>
            <tempo><{tick}>2400</{tick}><{bpm}>12000</{bpm}></tempo></masterTrack>
            <vsTrack><{part}><{tick}>1920</{tick}><note><{tick}>0</{tick}><{dur}>960</{dur}>
            <{pitch}>60</{pitch}><{lyric}>la</{lyric}></note></{part}></vsTrack></vsq{version}>'''
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "song.vsqx"
                path.write_text(xml, encoding="utf-8")
                project, notes = load_score(path)
                self.assertEqual(notes, [(0, 1.5, 60)])
                self.assertEqual(project["tracks"][0]["mainGroup"]["notes"][0]["duration"], 2 * QUARTER)

    def test_svp_reference_offset_and_overlap(self):
        project = dict(time=dict(tempo=[dict(position=0, bpm=120)]), tracks=[dict(
            mainRef=dict(blickOffset=QUARTER, pitchOffset=2),
            mainGroup=dict(notes=[dict(onset=0, duration=QUARTER, pitch=60, lyrics="la")]))])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "song.svp"
            path.write_text(json.dumps(project), encoding="utf-8")
            _, notes = load_score(path)
            self.assertEqual(notes, [(.5, 1, 62)])
            project["tracks"][0]["mainGroup"]["notes"] *= 2
            path.write_text(json.dumps(project), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重叠"):
                load_score(path)

    def test_alignment_offset_and_truncation(self):
        audio = np.zeros(6000, np.float32)
        audio[3000:4000] = .1
        notes = [(1, 2, 60)]
        _, errors, _ = alignment(audio, 1000, notes, 0)
        self.assertTrue(errors)
        _, errors, metrics = alignment(audio, 1000, notes, 2)
        self.assertFalse(errors)
        self.assertGreater(metrics["audio_coverage"], .99)
        _, errors, _ = alignment(audio, 1000, notes, 5)
        self.assertTrue(errors)

    def test_multiple_active_tracks_rejected(self):
        track = dict(mainGroup=dict(notes=[dict(onset=0, duration=QUARTER, pitch=60, lyrics="la")]))
        project = dict(time=dict(tempo=[dict(position=0, bpm=120)]), tracks=[track, dict(track, mixer=dict(mute=True))])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "song.svp"
            path.write_text(json.dumps(project), encoding="utf-8")
            self.assertEqual(len(load_score(path)[1]), 1)
            project["tracks"][1]["mixer"]["mute"] = False
            path.write_text(json.dumps(project), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "一条人声轨"):
                load_score(path)

    def test_crop_preserves_time_origin(self):
        np.testing.assert_array_equal(crop(np.array([1, 2, 3]), -2, 4), [0, 0, 1, 2])
        np.testing.assert_array_equal(crop(np.array([1, 2, 3]), 1, 4), [2, 3, 0, 0])


if __name__ == "__main__":
    unittest.main()
