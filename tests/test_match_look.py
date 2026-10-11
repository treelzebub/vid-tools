"""Tests for match_look.py.

Run from the repo root:   python3 -m unittest discover -s _Scripts/tests -v
Face detection can't be exercised on synthetic frames, so the end-to-end tests replace
`measure` with a function that returns a fixed skin color per video size; everything else
(sampling, outlier rejection, the correction, LUT/preview/report writing, caching) is real.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("match_look", os.path.join(SCRIPTS, "match_look.py"))
ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ml)

REF_SKIN = np.array([182.0, 116.0, 94.0])
TARGET_SKIN = np.array([222.0, 178.0, 166.0])
NO_TONE = dict(black_point=0.0, highlight_cap=255.0)  # leaves only the exposure/white-balance gains


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_lines(path):
    with open(path) as f:
        return f.read().split("\n")


def corr(target=TARGET_SKIN, ref=REF_SKIN, **kw):
    return ml.Correction(target, ref, **kw)


class TestColorMath(unittest.TestCase):
    def test_srgb_round_trip(self):
        x = np.linspace(0, 255, 256)
        np.testing.assert_allclose(ml.to_srgb(ml.to_lin(x)), x, atol=1e-6)

    def test_known_values(self):
        self.assertAlmostEqual(float(ml.to_lin(255)), 1.0)
        self.assertAlmostEqual(float(ml.to_lin(0)), 0.0)
        self.assertAlmostEqual(float(ml.to_lin(128)), 0.2158, places=3)

    def test_delta_e(self):
        c = np.array([200.0, 120.0, 90.0])
        self.assertEqual(ml.delta_e(c, c), 0.0)
        self.assertAlmostEqual(ml.delta_e(c, REF_SKIN), ml.delta_e(REF_SKIN, c))
        L, a, b = ml.lab(np.array([255.0, 255.0, 255.0]))
        self.assertAlmostEqual(L, 100.0, places=0)
        self.assertAlmostEqual(a, 0.0, places=0)
        self.assertAlmostEqual(b, 0.0, places=0)


class TestCorrection(unittest.TestCase):
    def test_identity_when_target_equals_reference(self):
        c = corr(REF_SKIN, REF_SKIN, **NO_TONE)
        self.assertAlmostEqual(c.exposure, 1.0)
        np.testing.assert_allclose(c.wb_log, 0.0, atol=1e-9)
        mid = np.array([[40.0, 90.0, 140.0], [100.0, 100.0, 100.0], [10.0, 200.0, 60.0]])
        np.testing.assert_allclose(c.transform(mid, 1.0), mid, atol=1e-6 * 255)

    def test_exposure_closes_the_requested_share_of_the_gap_in_log_domain(self):
        for pull in (0.0, 0.3, 0.6, 1.0):
            c = corr(skin_pull=pull, chroma_pull=0.0, **NO_TONE)
            y_t = float(ml.to_lin(TARGET_SKIN) @ ml.W)
            y_r = float(ml.to_lin(REF_SKIN) @ ml.W)
            out_y = float(ml.to_lin(c.transform(TARGET_SKIN, 0.0)) @ ml.W)
            self.assertAlmostEqual(out_y, y_t * (y_r / y_t) ** pull, places=6, msg=f"pull={pull}")

    def test_white_balance_keeps_skin_brightness(self):
        c = corr(**NO_TONE)
        y0 = float(ml.to_lin(c.transform(TARGET_SKIN, 0.0)) @ ml.W)
        for alpha in (0.4, 1.0):
            y = float(ml.to_lin(c.transform(TARGET_SKIN, alpha)) @ ml.W)
            self.assertAlmostEqual(y, y0, places=6, msg=f"alpha={alpha}")

    def test_white_balance_moves_skin_toward_the_reference(self):
        c = corr(**NO_TONE)
        d = [ml.delta_e(c.transform(TARGET_SKIN, a), REF_SKIN) for a in (0.0, 0.4, 1.0)]
        self.assertGreater(ml.delta_e(TARGET_SKIN, REF_SKIN), d[0])
        self.assertGreater(d[0], d[1])
        self.assertGreater(d[1], d[2])

    def test_exposure_only_keeps_grays_neutral(self):
        c = corr(**NO_TONE)
        grays = np.array([[g, g, g] for g in (20.0, 80.0, 140.0, 200.0)])
        out = c.transform(grays, 0.0)
        np.testing.assert_allclose(out[:, 0], out[:, 1], atol=1e-6)
        np.testing.assert_allclose(out[:, 1], out[:, 2], atol=1e-6)

    def test_gains_are_capped(self):
        c = corr(np.array([250.0, 240.0, 235.0]), np.array([120.0, 60.0, 30.0]), max_wb=1.3)
        self.assertTrue(c.wb_clamped)
        self.assertLessEqual(float(np.max(np.abs(c.wb_log))), np.log(1.3) + 1e-12)
        mild = corr(np.array([200.0, 150.0, 130.0]), np.array([205.0, 152.0, 128.0]), max_wb=1.3)
        self.assertFalse(mild.wb_clamped)

    def test_output_is_bounded_and_monotone_even_when_brightening(self):
        c = corr(np.array([90.0, 60.0, 50.0]), np.array([200.0, 150.0, 130.0]))  # a big brightening
        self.assertGreater(c.stops, 0.5)
        ramp = np.array([[g, g, g] for g in np.linspace(0, 255, 256)])
        for alpha in (0.0, 1.0):
            out = c.transform(ramp, alpha)
            self.assertLessEqual(float(out.max()), c.highlight_cap + 1e-9)
            self.assertGreaterEqual(float(out.min()), 0.0)
            self.assertTrue(np.all(np.diff(out[:, 1]) >= -1e-9), "gray ramp must not invert")

    def test_black_point_and_highlight_cap(self):
        c = corr(REF_SKIN, REF_SKIN, black_point=10.0, highlight_cap=250.0)  # no exposure change
        low = c.transform(np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]), 0.0)
        np.testing.assert_allclose(low, 0.0)
        white = c.transform(np.array([[255.0, 255.0, 255.0]]), 0.0)
        self.assertAlmostEqual(float(white.max()), 250.0, places=6)


class TestLut(unittest.TestCase):
    def test_cube_file_layout(self):
        c = corr()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.cube")
            ml.write_lut(p, c, 1.0, "a title", size=5)
            lines = read_lines(p)
        self.assertEqual(lines[0], 'TITLE "a title"')
        self.assertEqual(lines[1], "LUT_3D_SIZE 5")
        rows = np.array([[float(v) for v in l.split()] for l in lines[2:] if l])
        self.assertEqual(rows.shape, (125, 3))
        self.assertTrue(((rows >= 0) & (rows <= 1)).all())
        grid = np.linspace(0, 255, 5)
        # R varies fastest: row 1 is (R=64, G=0, B=0), row 5 is (R=0, G=64, B=0), row 25 is (0, 0, 64)
        for row, rgb in ((1, (grid[1], 0, 0)), (5, (0, grid[1], 0)), (25, (0, 0, grid[1])), (124, (255, 255, 255))):
            np.testing.assert_allclose(rows[row], c.transform(np.array(rgb, float), 1.0) / 255, atol=1e-6)


class TestFramesAndMeasure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            raise unittest.SkipTest("ffmpeg/ffprobe not installed")
        cls.tmp = tempfile.mkdtemp(prefix="matchlook-tests-")
        cls.video = os.path.join(cls.tmp, "t.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=25:duration=4",
                        "-pix_fmt", "yuv420p", cls.video], check=True)
        cls.flat = os.path.join(cls.tmp, "flat.png")
        Image.fromarray(np.full((90, 160, 3), 128, np.uint8)).save(cls.flat)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_frame_from_video_and_png(self):
        self.assertEqual(ml.frame(self.video, 1.0).shape, (90, 160, 3))
        self.assertEqual(ml.frame(self.flat).shape, (90, 160, 3))

    def test_frame_past_the_end_is_none(self):
        self.assertIsNone(ml.frame(self.video, 3600))

    def test_no_face_in_a_flat_image(self):
        self.assertIsNone(ml.measure(np.full((300, 300, 3), 128, np.uint8)))

    def test_collect_exits_when_too_few_faces(self):
        with self.assertRaises(SystemExit) as cm:
            ml.collect(self.video, [0.5, 1.0, 1.5, 2.0])
        self.assertIn("too few usable frames", str(cm.exception))

    def test_collect_rejects_brightness_outliers(self):
        skins = iter([np.array([180.0, 120.0, 100.0])] * 6 + [np.array([240.0, 230.0, 220.0])]  # glare frame
                     + [np.array([181.0, 121.0, 99.0])] * 3)
        with mock.patch.object(ml, "measure", lambda img: next(skins)):
            r = ml.collect(self.video, [0.2 * i for i in range(10)])
        self.assertEqual((r["tried"], r["found"], r["used"]), (10, 10, 9))
        self.assertEqual(len(r["usable_times"]), 9)
        np.testing.assert_allclose(r["skin"], [180.0, 120.0, 100.0], atol=1.0)

    def test_collect_median_is_not_dragged_by_one_bad_frame(self):
        vals = [np.array([182.0, 118.0, 96.0])] * 7 + [np.array([60.0, 40.0, 30.0])]
        it = iter(vals)
        with mock.patch.object(ml, "measure", lambda img: next(it)):
            r = ml.collect(self.video, [0.1 * i for i in range(8)])
        self.assertEqual(r["used"], 7)
        np.testing.assert_allclose(r["skin"], [182.0, 118.0, 96.0])


class TestEndToEnd(unittest.TestCase):
    """Real sampling, LUT, preview, report, and cache code; skin color is faked by video size."""

    @classmethod
    def setUpClass(cls):
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            raise unittest.SkipTest("ffmpeg/ffprobe not installed")
        cls.tmp = tempfile.mkdtemp(prefix="matchlook-e2e-")
        mk = lambda name, size: subprocess.run(  # noqa: E731
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=25:duration=6",
             "-pix_fmt", "yuv420p", os.path.join(cls.tmp, name)], check=True)
        mk("ref.mp4", "320x180")
        mk("target.mp4", "160x90")
        cls.ref = os.path.join(cls.tmp, "ref.mp4")
        cls.target = os.path.join(cls.tmp, "target.mp4")
        cls.still = os.path.join(cls.tmp, "ref.png")
        Image.fromarray(np.full((180, 320, 3), 100, np.uint8)).save(cls.still)
        cls.calls = []

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def fake_measure(self, ref_skin=REF_SKIN, target_skin=TARGET_SKIN, usable_target=None):
        counter = {"target": 0}

        def fake(img):
            is_ref = img.shape[1] == 320
            self.calls.append("ref" if is_ref else "target")
            if is_ref:
                return ref_skin
            counter["target"] += 1
            if usable_target is not None and counter["target"] > usable_target:
                return None
            return target_skin
        return fake

    def run_main(self, *extra, fake=None, out=None):
        out = out or tempfile.mkdtemp(dir=self.tmp)
        self.calls.clear()
        with mock.patch.object(ml, "measure", fake or self.fake_measure()):
            ml.main(["--reference", self.ref, *extra, self.target, out])
        return out

    def test_outputs_and_report(self):
        out = self.run_main("--name", "L9")
        self.assertEqual(sorted(os.listdir(out)), sorted([
            "L9 match.cube", "L9 match mild.cube", "L9 match neutral.cube",
            "L9 match-preview.png", "L9 match-report.json"]))
        r = read_json(os.path.join(out, "L9 match-report.json"))
        de = r["skin_delta_e76_to_reference"]
        self.assertGreater(de["raw"], de["neutral"])
        self.assertGreater(de["neutral"], de["mild"])
        self.assertGreater(de["mild"], de["full"])
        self.assertLess(r["gains"]["exposure_stops"], 0)  # the target skin is brighter than the reference
        self.assertEqual(r["target_frames"]["used"], 16)
        self.assertEqual(r["reference_frames"]["tried"], 40)
        self.assertEqual(r["lumetri_approx"]["temperature"], "warmer")
        self.assertEqual(r["name"], "L9")

    def test_name_defaults_to_the_target_file_name(self):
        out = self.run_main()
        self.assertIn("target match.cube", os.listdir(out))

    def test_preview_has_five_columns_and_three_rows(self):
        out = self.run_main("--name", "p")
        with Image.open(os.path.join(out, "p match-preview.png")) as im:
            w, h = im.size
        self.assertEqual((w, h), (5 * 160 // 3, 3 * 90 // 3))

    def test_lut_size_option(self):
        out = self.run_main("--name", "s", "--lut-size", "9")
        rows = [l for l in read_lines(os.path.join(out, "s match.cube"))[2:] if l]
        self.assertEqual(len(rows), 9 ** 3)

    def test_tuning_flags_change_the_result(self):
        a = read_json(os.path.join(self.run_main("--name", "a"), "a match-report.json"))
        b = read_json(os.path.join(self.run_main("--name", "b", "--skin-pull", "1.0", "--max-wb", "2.0"),
                                        "b match-report.json"))
        self.assertLess(b["gains"]["exposure_stops"], a["gains"]["exposure_stops"])
        self.assertNotEqual(a["gains"]["white_balance_rgb"], b["gains"]["white_balance_rgb"])

    def test_reference_times_are_used(self):
        self.run_main("--reference-times", "1,2,3,4", "--name", "t")
        self.assertEqual(self.calls.count("ref"), 4)
        self.assertEqual(self.calls.count("target"), 16)

    def test_cache_skips_remeasuring_the_reference(self):
        cache = os.path.join(self.tmp, "ref-cache.json")
        if os.path.exists(cache):
            os.remove(cache)
        self.run_main("--cache", cache, "--name", "c1")
        self.assertEqual(self.calls.count("ref"), 40)
        self.run_main("--cache", cache, "--name", "c2")
        self.assertEqual(self.calls.count("ref"), 0)
        out = self.run_main("--cache", cache, "--name", "c3")
        r = read_json(os.path.join(out, "c3 match-report.json"))
        self.assertEqual(r["reference_frames"]["used"], 40)
        # changing the reference invalidates it
        os.utime(self.ref, (1_000_000, 1_000_000))
        self.run_main("--cache", cache, "--name", "c4")
        self.assertEqual(self.calls.count("ref"), 40)

    def test_still_image_reference(self):
        out = tempfile.mkdtemp(dir=self.tmp)
        self.calls.clear()
        with mock.patch.object(ml, "measure", self.fake_measure()):
            ml.main(["--reference", self.still, "--name", "still", self.target, out])
        r = read_json(os.path.join(out, "still match-report.json"))
        self.assertEqual(r["reference_frames"]["used"], 1)
        self.assertEqual(self.calls.count("ref"), 1)
        self.assertTrue(os.path.isfile(os.path.join(out, "still match-preview.png")))

    def test_weak_target_is_reported(self):
        out = self.run_main("--name", "w", fake=self.fake_measure(usable_target=4))
        r = read_json(os.path.join(out, "w match-report.json"))
        self.assertEqual(r["target_frames"]["used"], 4)
        self.assertTrue(any("only 4 of 16" in w for w in r["warnings"]), r["warnings"])

    def test_unstable_target_brightness_is_reported(self):
        vals = iter([np.array([c, c * 0.65, c * 0.55]) for c in
                     (120, 200, 140, 220, 130, 210, 125, 205, 135, 215, 128, 208, 122, 202, 138, 218)])

        def fake(img):
            return REF_SKIN if img.shape[1] == 320 else next(vals)
        out = self.run_main("--name", "d", fake=fake)
        r = read_json(os.path.join(out, "d match-report.json"))
        self.assertTrue(any("varies by" in w for w in r["warnings"]), r["warnings"])

    def test_identical_skin_needs_no_correction(self):
        out = self.run_main("--name", "same", fake=self.fake_measure(target_skin=REF_SKIN))
        r = read_json(os.path.join(out, "same match-report.json"))
        self.assertEqual(r["gains"]["exposure_stops"], 0.0)
        self.assertEqual(r["gains"]["white_balance_rgb"], [1.0, 1.0, 1.0])
        self.assertEqual(r["warnings"], [])

    def test_errors(self):
        with self.assertRaises(SystemExit) as cm:
            ml.main(["--reference", "/nope.mp4", self.target, self.tmp])
        self.assertIn("reference not found", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            ml.main(["--reference", self.ref, "/nope.mp4", self.tmp])
        self.assertIn("target not found", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            ml.main(["--reference", self.ref, self.still, self.tmp])
        self.assertIn("target must be a video", str(cm.exception))
        with self.assertRaises(SystemExit):  # argparse: --reference is required
            ml.main([self.target, self.tmp])

    def test_output_directory_is_created(self):
        out = os.path.join(self.tmp, "new", "nested")
        self.run_main("--name", "n", out=out)
        self.assertTrue(os.path.isfile(os.path.join(out, "n match.cube")))


if __name__ == "__main__":
    unittest.main()
