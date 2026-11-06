import unittest

from opensv_train.segments import annotated_segments


class SegmentationTests(unittest.TestCase):
    def condition(self):
        return dict(start=-.1, counts=[20, 40, 20, 40, 20],
                    phones=[float(i) for i in range(5) for _ in range(108)],
                    languages=[0, 1, 2, 3, 4], noise=list(range(560)), log_f0=list(range(140)))

    def test_deleted_note_gap_and_padding_are_excluded(self):
        result = annotated_segments([self.condition()], [(0, .2, 60), (.3, .5, 62)])
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]["start"], 0)
        self.assertAlmostEqual(result[1]["start"], .3)
        self.assertEqual([s["counts"] for s in result], [[40], [40]])
        self.assertEqual([s["languages"] for s in result], [[1], [3]])
        self.assertEqual(result[1]["noise"], list(range(320, 480)))
        self.assertEqual(result[0]["log_f0"], list(range(20, 60)))
        self.assertEqual(result[1]["phones"], [3.] * 108)

    def test_adjacent_notes_stay_together(self):
        result = annotated_segments([self.condition()], [(0, .2, 60), (.2, .5, 62)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["counts"], [40, 20, 40])

    def test_fractional_boundaries_keep_only_full_frames(self):
        result = annotated_segments([self.condition()], [(.001, .199, 60)])[0]
        self.assertAlmostEqual(result["start"], .005)
        self.assertEqual(sum(result["counts"]), 38)
        self.assertAlmostEqual(result["start"] + sum(result["counts"]) * .005, .195)

    def test_unannotated_phrase_is_excluded(self):
        self.assertEqual(annotated_segments([self.condition()], [(2, 3, 60)]), [])

    def test_112_channels_preserve_phoneme_rows(self):
        condition = self.condition()
        condition["phones"] = [float(i) for i in range(5) for _ in range(112)]
        result = annotated_segments([condition], [(.3, .5, 62)])[0]
        self.assertEqual(result["phones"], [3.] * 112)
        self.assertEqual(result["counts"], [40])

    def test_audio_crop_keeps_absolute_position_and_offset(self):
        import numpy as np
        from opensv_train.prepare import crop
        audio = np.arange(2000, dtype=np.float32)
        segments = annotated_segments([self.condition()], [(0, .2, 60), (.3, .5, 62)])
        crops = [crop(audio, round((s["start"] + .1) * 1000), sum(s["counts"]) * 5)
                 for s in segments]
        np.testing.assert_array_equal(crops[0], audio[100:300])
        np.testing.assert_array_equal(crops[1], audio[400:600])

    def test_unannotated_singing_does_not_reject_valid_notes(self):
        import numpy as np
        from opensv_train.prepare import alignment
        _, errors, _ = alignment(np.ones(10000), 1000, [(1, 2, 60)], 0)
        self.assertEqual(errors, [])
        audio = np.zeros(10000)
        audio[5000:6000] = 1
        _, errors, _ = alignment(audio, 1000, [(1, 2, 60)], 0)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
