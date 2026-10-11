"""Tests for make_prproj.py and new_lecture.sh.

Run from the repo root:   python3 -m unittest discover -s _Scripts/tests -v
Needs ffmpeg/ffprobe (a few 2-second clips are generated). No Premiere, no real project files.
"""
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import itertools
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(__file__))
import fixture as fx  # noqa: E402

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.environ.get("MAKE_PRPROJ", os.path.join(SCRIPTS, "make_prproj.py"))  # override to test a variant
CONFIG = os.path.join(SCRIPTS, "prproj.config.json")
WRAPPER = os.path.join(SCRIPTS, "new_lecture.sh")

# the keep rules the real config uses, for tests that don't go through the config file
KEEP = [
    "V1= - Gallery View\\.mp4$",
    "V2= - Speaker View\\.mp4$",
    "V5= - Shared Screen with Speaker View\\.mp4$",
    "V6= - Shared Screen\\.mp4$",
    "A1= - Shared Screen with Speaker View\\.mp4$",
]
KEEP_ARGS = [a for k in KEEP for a in ("--keep", k)]


# ---------------------------------------------------------------- helpers
def load(path):
    text = gzip.open(path).read().decode("utf-8")
    root = ET.fromstring(re.sub(r"&#(\d+);", r"\1", text).encode("utf-8"))
    ids = {e.get(k): e for e in root for k in ("ObjectID", "ObjectUID") if e.get(k)}
    return root, ids


def raw(path):
    return gzip.open(path).read().decode("utf-8")


def tracks(path, seqname="Sequence 01"):
    """{'V': [track,...], 'A': [...]}; track = dict(clips=[...], locked, transitions)."""
    root, ids = load(path)
    seq = next(e for e in root if e.tag == "Sequence" and e.findtext("Name") == seqname)
    out = {"V": [], "A": []}
    for g in seq.findall(".//TrackGroups/TrackGroup"):
        grp = ids[g.find("Second").get("ObjectRef")]
        key = "V" if grp.tag == "VideoTrackGroup" else "A"
        for t in grp.findall(".//Tracks/Track"):
            tr = ids[t.get("ObjectURef")]
            clips = []
            for ti in tr.findall(".//ClipItems/TrackItems/TrackItem"):
                it = ids[ti.get("ObjectRef")]
                sc = ids[it.find("ClipTrackItem/SubClip").get("ObjectRef")]
                clip = ids[sc.find("Clip").get("ObjectRef")]
                src = ids[clip.find(".//Source").get("ObjectRef")]
                media = ids[src.find("MediaSource/Media").get("ObjectURef")]
                clips.append(dict(
                    name=sc.findtext("Name"), media=media.findtext("Title"), item=it, clip=clip,
                    start=int(it.findtext(".//Start")), end=int(it.findtext(".//End")),
                    inp=int(clip.findtext(".//InPoint")), out=int(clip.findtext(".//OutPoint"))))
            out[key].append(dict(
                clips=clips, locked=tr.find(".//IsLocked") is not None,
                transitions=len(tr.findall(".//TransitionItems/TrackItems/TrackItem"))))
    return out


def stream_min_seconds(path):
    o = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", path], text=True))
    return min(float(s["duration"]) for s in o["streams"])


def chain_effects(root, ids):
    """MatchNames of effects actually attached to clips (orphaned components stay in the file)."""
    return sorted(ids[c.get("ObjectRef")].findtext(".//MatchName")
                  for ch in root if ch.tag == "VideoComponentChain"
                  for c in ch.find("ComponentChain/Components"))


_OUT_COUNTER = itertools.count(1)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            raise unittest.SkipTest("ffmpeg/ffprobe not installed")
        cls.tmp = tempfile.mkdtemp(prefix="prproj-tests-")
        cls.template_root = os.path.join(cls.tmp, "Course", "Lecture 4")
        cls.template = os.path.join(cls.tmp, "template.prproj")
        fx.build(cls.template, cls.template_root, n=4)
        cls.media = os.path.join(cls.tmp, "proj", "Sources")
        cls.paths = fx.make_media(cls.media, 7)
        cls.proj_root = os.path.join(cls.tmp, "proj")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def outpath(self):
        d = os.path.join(self.proj_root, "Premiere")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"out{next(_OUT_COUNTER)}.prproj")

    def run_script(self, *args, out=None, expect_ok=True):
        out = out or self.outpath()
        p = subprocess.run([sys.executable, SCRIPT, *args, out], capture_output=True, text=True)
        text = p.stdout + p.stderr
        if expect_ok:
            self.assertEqual(p.returncode, 0, text)
            self.assertTrue(os.path.isfile(out), text)
        else:
            self.assertNotEqual(p.returncode, 0, "expected failure but it succeeded:\n" + text)
            self.assertFalse(os.path.exists(out), "failed run must not leave an output file")
        return out, text

    def build(self, *extra, media=None, template=None, out=None):
        return self.run_script(
            "--template", template or self.template, "--media-dir", media or self.media,
            "--sequence", "Sequence 01", *KEEP_ARGS, *extra, out=out)[0]

    def expected_d(self, media_paths=None, frame=fx.FPS25):
        secs = min(stream_min_seconds(p) for p in (media_paths or self.paths.values()))
        return int(secs * fx.TICKS_PER_SEC / frame) * frame


# ---------------------------------------------------------------- timeline
class TestTimeline(Base):
    def test_kept_tracks_have_one_full_length_clip(self):
        out = self.build()
        D = self.expected_d()
        t = tracks(out)
        want_v = {0: fx.GAL, 1: fx.SPK, 4: fx.SSWSV, 5: fx.SS}
        for i, tr in enumerate(t["V"]):
            if i in want_v:
                self.assertEqual(len(tr["clips"]), 1, f"V{i + 1}")
                c = tr["clips"][0]
                self.assertEqual(c["media"], fx.media_name(want_v[i], 7))
                self.assertEqual((c["start"], c["end"], c["inp"], c["out"]), (0, D, 0, D), f"V{i + 1}")
            else:
                self.assertEqual(tr["clips"], [], f"V{i + 1} should be bare")
        self.assertEqual([len(a["clips"]) for a in t["A"]], [1, 0])
        self.assertEqual(t["A"][0]["clips"][0]["media"], fx.media_name(fx.SSWSV, 7))
        self.assertEqual(t["A"][0]["clips"][0]["end"], D)

    def test_keep_picks_by_regex_not_by_position(self):
        # V2's first clip in the template is Gallery View; the Speaker View clip must win
        t = tracks(self.build())
        self.assertEqual(t["V"][1]["clips"][0]["media"], fx.media_name(fx.SPK, 7))

    def test_transitions_removed(self):
        t = tracks(self.build())
        self.assertEqual([tr["transitions"] for tr in t["V"]], [0] * 6)
        # the kept V1 clip must have lost both transition refs
        v1 = t["V"][0]["clips"][0]["item"]
        self.assertIsNone(v1.find("ClipTrackItem/HeadTransition"))
        self.assertIsNone(v1.find("ClipTrackItem/TailTransition"))

    def test_sequence_view_state_reset(self):
        out = self.build()
        root, _ = load(out)
        D = str(self.expected_d())
        seq = next(e for e in root if e.tag == "Sequence" and e.findtext("Name") == "Sequence 01")
        p = seq.find("Node/Properties")
        self.assertEqual(p.findtext("MZ.InPoint"), "0")
        self.assertEqual(p.findtext("MZ.EditLine"), "0")
        self.assertEqual(p.findtext("MZ.OutPoint"), D)
        self.assertEqual(p.findtext("MZ.WorkOutPoint"), D)
        self.assertEqual(p.findtext("Monitor.ProgramZoomOut"), D)

    def test_other_sequences_untouched(self):
        t = tracks(self.build(), "1080p 23.98")
        self.assertEqual(t, {"V": [], "A": []})

    def test_frame_duration_comes_from_the_sequence(self):
        tpl = os.path.join(self.tmp, "template30.prproj")
        fx.build(tpl, self.template_root, n=4, frame=fx.FPS30)
        # a length that is a whole number of 30 fps frames but not of 25 fps frames,
        # so a hardcoded 25 fps would give a different answer
        d = os.path.join(self.tmp, "odd", "Sources")
        paths = fx.make_media(d, 7, seconds=2.1)
        out = self.build(template=tpl, media=d)
        c = tracks(out)["V"][0]["clips"][0]
        self.assertEqual(c["end"], self.expected_d(paths.values(), frame=fx.FPS30))
        self.assertEqual(c["end"] % fx.FPS30, 0)
        self.assertNotEqual(c["end"], self.expected_d(paths.values(), frame=fx.FPS25))

    def test_links_and_groups_filtered_to_survivors(self):
        root, ids = load(self.build())
        links = [e for e in root if e.tag == "Link"]
        groups = [e for e in root if e.tag == "PersistentGroup"]
        self.assertEqual(len(links), 1, "link that lost a member must be dropped, the intact one kept")
        self.assertEqual(len(links[0].findall("TrackItemGroup/TrackItems/TrackItem")), 2)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].findall("TrackItemGroup/TrackItems/TrackItem")), 2)
        for m in links[0].findall(".//TrackItem") + groups[0].findall(".//TrackItem"):
            self.assertIn(m.get("ObjectRef"), ids)

    def test_markers_cleared(self):
        root, _ = load(self.build())
        for mk in (e for e in root if e.tag == "Markers"):
            self.assertEqual(len(mk.find("Markers")), 0)


# ---------------------------------------------------------------- media
class TestMedia(Base):
    def test_media_retargeted(self):
        out = self.build()
        root, ids = load(out)
        D = str(self.expected_d())
        media = [e for e in root if e.tag == "Media"]
        new = [m for m in media if m.findtext("Title").endswith(".mp4")]
        self.assertEqual(len(new), 4)
        for m in new:
            title = m.findtext("Title")
            self.assertIn("Lecture 7", title)
            self.assertNotIn("Lecture 4", title)
            want = os.path.join(self.media, title)
            self.assertEqual(m.findtext("ActualMediaFilePath"), want)
            self.assertEqual(m.findtext("FilePath"), want)
            for rp in m.findall("RelativePath"):
                self.assertEqual(rp.text, os.path.relpath(want, os.path.dirname(out)))
                self.assertEqual(rp.text, "../Sources/" + title)
            for tag in ("AudioStream", "VideoStream"):
                self.assertEqual(ids[m.find(tag).get("ObjectRef")].findtext("Duration"), D)
            uid = m.get("ObjectUID")
            for e in root:
                if e.tag in ("VideoMediaSource", "AudioMediaSource") and \
                        e.find("MediaSource/Media").get("ObjectURef") == uid:
                    self.assertEqual(e.findtext("OriginalDuration"), D)

    def test_no_references_to_template_media_files_remain(self):
        text = raw(self.build())
        for kind in fx.KINDS:
            self.assertNotIn(fx.media_name(kind, 4), text)
        # only the template's directory (cache, export, slide paths) is left; --project-root moves that
        self.assertNotIn("Lecture 4", raw(self.build("--project-root", self.proj_root)))

    def test_cache_paths_follow_the_new_media(self):
        root, ids = load(self.build())
        for e in root:
            if e.tag == "VideoStream" and e.findtext("ConformedAudioPath"):
                if "Lecture 7" in e.findtext("ConformedAudioPath"):
                    self.assertIn("Lecture 7", e.findtext("PeakFilePath"))

    def test_digit_runs_differ_in_length(self):
        # template says "Lecture 4", the new files say "Lecture 12"
        d = os.path.join(self.tmp, "twelve", "Sources")
        paths = fx.make_media(d, 12)
        out = self.build(media=d)
        titles = {m.findtext("Title") for m in load(out)[0] if m.tag == "Media"}
        self.assertIn(fx.media_name(fx.SPK, 12), titles)
        self.assertEqual(self.expected_d(paths.values()), tracks(out)["V"][0]["clips"][0]["end"])

    def test_stills_dropped_from_bin_but_not_video_items(self):
        root, ids = load(self.build())
        names = [e.findtext(".//Name") for e in root if e.tag == "ClipProjectItem"
                 and e.get("ObjectID") in {i.get("ObjectRef") for i in
                                           next(x for x in root if x.tag == "RootProjectItem").findall(".//Item")}]
        self.assertFalse([n for n in names if n.endswith(".png")], names)
        self.assertEqual(len([n for n in names if n.endswith(".mp4")]), 4)

    def test_orphans_are_kept(self):
        # Premiere rejects projects whose orphaned objects were deleted
        src_root, _ = load(self.template)
        out_root, _ = load(self.build())
        keep_tags = {"VideoClipTrackItem", "AudioClipTrackItem", "SubClip", "VideoClip", "AudioClip",
                     "VideoFilterComponent", "VideoComponentChain", "VideoTransitionTrackItem", "Markers"}
        count = lambda r: sum(1 for e in r if e.tag in keep_tags)  # noqa: E731
        self.assertEqual(count(out_root), count(src_root))


# ---------------------------------------------------------------- effects / locks / serialization
class TestEffectsAndSerialization(Base):
    def test_effects_untouched_by_default(self):
        root, ids = load(self.build())
        names = chain_effects(root, ids)
        self.assertIn("AE.ADBE Lumetri", names)
        self.assertIn("AE.ADBE Sharpen", names)

    def test_strip_effect(self):
        root, ids = load(self.build("--strip-effect", "AE.ADBE Lumetri", "--strip-effect", "AE.ADBE Sharpen"))
        for chain in (e for e in root if e.tag == "VideoComponentChain"):
            comps = chain.find("ComponentChain/Components")
            kinds = [ids[c.get("ObjectRef")].findtext(".//MatchName") for c in comps]
            self.assertNotIn("AE.ADBE Lumetri", kinds)
            self.assertNotIn("AE.ADBE Sharpen", kinds)
            self.assertIn("AE.ADBE Motion", kinds)
            self.assertEqual([c.get("Index") for c in comps], [str(i) for i in range(len(comps))])

    def test_strip_one_effect_only(self):
        root, ids = load(self.build("--strip-effect", "AE.ADBE Sharpen"))
        kinds = {ids[c.get("ObjectRef")].findtext(".//MatchName")
                 for ch in root if ch.tag == "VideoComponentChain"
                 for c in ch.find("ComponentChain/Components")}
        self.assertNotIn("AE.ADBE Sharpen", kinds)
        self.assertIn("AE.ADBE Lumetri", kinds)

    def test_locks_kept_unless_unlock(self):
        self.assertTrue(all(t["locked"] for t in tracks(self.build())["V"]))
        t = tracks(self.build("--unlock"))
        self.assertFalse(any(tr["locked"] for tr in t["V"] + t["A"]))

    def test_serialization_fidelity(self):
        out = self.build()
        text = raw(out)
        self.assertTrue(text.startswith('<?xml version="1.0" encoding="UTF-8" ?>\n<PremiereData Version="3">'))
        self.assertTrue(text.endswith("</PremiereData>\n"))
        self.assertIn("&#10;", text)  # numeric entities written back verbatim
        self.assertIn("&#9;", text)
        self.assertIn("&lt;?xml", text)  # escaped markup stays escaped
        self.assertNotIn(" />", text)  # Premiere's self-closing style
        self.assertEqual(text.count("&#10;"), raw(self.template).count("&#10;"))
        ET.fromstring(re.sub(r"&#(\d+);", r"\1", text).encode())  # still well-formed

    def test_index_attributes_contiguous(self):
        root, _ = load(self.build("--strip-effect", "AE.ADBE Lumetri"))
        for parent in root.iter():
            idx = [c.get("Index") for c in parent if c.get("Index") is not None]
            self.assertEqual(idx, [str(i) for i in range(len(idx))], parent.tag)


# ---------------------------------------------------------------- failure modes
class TestFailures(Base):
    def fails(self, *args, msg, **kw):
        _, text = self.run_script(*args, expect_ok=False, **kw)
        self.assertIn(msg, text)

    def base_args(self, *keep, media=None):
        return ["--template", self.template, "--media-dir", media or self.media,
                "--sequence", "Sequence 01", *keep]

    def test_keep_on_missing_track(self):
        self.fails(*self.base_args("--keep", "V9=foo"), msg="only 6 V tracks")

    def test_keep_matching_nothing(self):
        self.fails(*self.base_args("--keep", "V6=zzz"), msg="matched no clip")

    def test_keep_malformed(self):
        self.fails(*self.base_args("--keep", "banana"), msg="--keep expects")

    def test_keep_matching_unpaired_media(self):
        self.fails(*self.base_args("--keep", "V4=slide-01"), msg="no counterpart")

    def test_two_template_media_to_one_file(self):
        d = os.path.join(self.tmp, "stray", "Sources")
        os.makedirs(d)
        for p in self.paths.values():
            shutil.copy(p, d)
        shutil.copy(self.paths[fx.SPK], os.path.join(d, "slide-7.png"))
        self.fails(*self.base_args(media=d), msg="would replace both")

    def test_nothing_pairs(self):
        d = os.path.join(self.tmp, "empty")
        os.makedirs(d)
        self.fails(*self.base_args(media=d), msg="pairs with any template media")

    def test_ambiguous_pairing(self):
        d = os.path.join(self.tmp, "ambig", "Sources")
        fx.make_media(d, 7, seconds=1, kinds=[fx.GAL])
        shutil.copy(os.path.join(d, fx.media_name(fx.GAL, 7)), os.path.join(d, fx.media_name(fx.GAL, 8)))
        self.fails(*self.base_args(media=d), msg="ambiguous")

    def test_unequal_lengths_refused_then_allowed(self):
        d = os.path.join(self.tmp, "unequal", "Sources")
        os.makedirs(d)
        for kind, p in self.paths.items():
            dst = os.path.join(d, os.path.basename(p))
            if kind == fx.GAL:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", p, "-t", "1", "-c", "copy", dst], check=True)
            else:
                shutil.copy(p, dst)
        self.fails(*self.base_args(*KEEP_ARGS, media=d), msg="differ by more than one frame")
        out = self.build("--allow-trim", media=d)
        short = stream_min_seconds(os.path.join(d, fx.media_name(fx.GAL, 7)))
        self.assertEqual(tracks(out)["V"][0]["clips"][0]["end"], int(short * 25) * fx.FPS25)

    def test_missing_files(self):
        self.fails("--template", "/nope.prproj", "--media-dir", self.media, msg="template not found")
        self.fails("--template", self.template, "--media-dir", "/nope", msg="media dir not found")

    def test_refuses_to_overwrite(self):
        out = self.build()
        p = subprocess.run([sys.executable, SCRIPT, "--template", self.template, "--media-dir", self.media,
                            "--sequence", "Sequence 01", out], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("refusing to overwrite", p.stdout + p.stderr)

    def test_sequence_must_be_chosen_when_ambiguous(self):
        self.fails("--template", self.template, "--media-dir", self.media, msg="choose the sequence")

    def test_unknown_sequence_lists_candidates(self):
        self.fails("--template", self.template, "--media-dir", self.media, "--sequence", "Nope",
                   msg="Sequence 01")

    def test_template_root_needs_project_root(self):
        self.fails(*self.base_args(*KEEP_ARGS), "--template-root", "/x", msg="no effect without --project-root")


# ---------------------------------------------------------------- project / template root
class TestRoots(Base):
    def test_project_root_rewrites_export_paths(self):
        out = self.build("--project-root", self.proj_root)
        text = raw(out)
        self.assertIn(f"{self.proj_root}/Premiere/Sequence 01_1.mp4", text)
        self.assertIn(f"{self.proj_root}/Premiere/template.prproj", text)
        self.assertNotIn(self.template_root, text)

    def test_without_project_root_export_paths_left_alone(self):
        text = raw(self.build())
        self.assertIn(f"{self.template_root}/Premiere/Sequence 01_1.mp4", text)

    def test_generated_project_as_template_needs_template_root(self):
        first = self.build("--project-root", self.proj_root)
        media8 = os.path.join(self.tmp, "l8", "Sources")
        fx.make_media(media8, 8)
        root8 = os.path.join(self.tmp, "l8")
        # media paths always follow --media-dir
        out = self.build("--project-root", root8, template=first, media=media8)
        # but the export paths still point at the old project: the template's media sits in Sources/
        self.assertIn(f"{self.proj_root}/Premiere/Sequence 01_1.mp4", raw(out))
        out2 = self.build("--project-root", root8, "--template-root", self.proj_root, template=first, media=media8)
        text = raw(out2)
        self.assertIn(f"{root8}/Premiere/Sequence 01_1.mp4", text)
        self.assertNotIn(self.proj_root, text)
        self.assertNotIn("Sources/Sources", text)


# ---------------------------------------------------------------- config file
class TestConfig(Base):
    def write_cfg(self, d, cfg):
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "cfg.json")
        with open(p, "w") as f:
            json.dump(cfg, f)
        return p

    def test_config_equals_command_line(self):
        cli = self.build("--strip-effect", "AE.ADBE Lumetri", "--unlock")
        cfg = self.write_cfg(os.path.join(self.tmp, "c1"), {
            "template": self.template, "media_dir": self.media, "sequence": "Sequence 01",
            "keep": {k.split("=")[0]: k.split("=", 1)[1] for k in KEEP},
            "strip_effect": ["AE.ADBE Lumetri"], "unlock": True})
        out, _ = self.run_script("--config", cfg)  # `out` positional comes last in run_script
        self.assertEqual(raw(out), raw(cli))

    def test_dashes_and_list_form_of_keep(self):
        cfg = self.write_cfg(os.path.join(self.tmp, "c2"), {
            "template": self.template, "media-dir": self.media, "sequence": "Sequence 01",
            "keep": KEEP, "unlock": True})
        t = tracks(self.run_script("--config", cfg)[0])
        self.assertEqual([len(x["clips"]) for x in t["V"]], [1, 1, 0, 0, 1, 1])
        self.assertFalse(any(x["locked"] for x in t["V"]))

    def test_relative_paths_resolve_against_the_config_file(self):
        d = os.path.join(self.tmp, "c3")
        os.makedirs(d, exist_ok=True)
        shutil.copy(self.template, os.path.join(d, "t.prproj"))
        os.symlink(self.media, os.path.join(d, "media"))
        cfg = self.write_cfg(d, {"template": "t.prproj", "media_dir": "media", "sequence": "Sequence 01",
                                 "keep": KEEP})
        env_cwd = tempfile.mkdtemp()
        out = self.outpath()
        p = subprocess.run([sys.executable, SCRIPT, "--config", cfg, out], cwd=env_cwd,
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_command_line_overrides_and_extends_config(self):
        cfg = self.write_cfg(os.path.join(self.tmp, "c4"), {
            "template": "/does/not/exist.prproj", "media_dir": self.media, "sequence": "nope",
            "keep": {"V6": "zzz"}})
        out, _ = self.run_script("--config", cfg, "--template", self.template, "--sequence", "Sequence 01",
                                 *[a for k in KEEP for a in ("--keep", k)])  # V6 entry replaced by the CLI's
        self.assertEqual(len(tracks(out)["V"][5]["clips"]), 1)

    def test_unknown_key(self):
        cfg = self.write_cfg(os.path.join(self.tmp, "c5"), {"bogus": 1})
        _, text = self.run_script("--config", cfg, expect_ok=False)
        self.assertIn("unknown key", text)

    def test_required_options_reported(self):
        cfg = self.write_cfg(os.path.join(self.tmp, "c6"), {"sequence": "Sequence 01"})
        _, text = self.run_script("--config", cfg, expect_ok=False)
        self.assertIn("template is required", text)


# ---------------------------------------------------------------- the real wrapper + real config
class TestNewLectureScript(Base):
    def test_end_to_end(self):
        root = os.path.join(self.tmp, "course")
        sd = os.path.join(root, "_Scripts")
        os.makedirs(sd)
        for f in ("new_lecture.sh", "make_prproj.py", "prproj.config.json"):
            shutil.copy(os.path.join(SCRIPTS, f), sd)
        tpl_root = os.path.join(root, "Lecture 05")
        os.makedirs(os.path.join(tpl_root, "Premiere"))
        tpl = os.path.join(tpl_root, "Premiere", "Visions of the End L05.prproj")
        fx.build(tpl, tpl_root, n=4)
        fx.make_media(os.path.join(root, "_Raw", "Lecture 9"), 9)

        p = subprocess.run(["bash", os.path.join(sd, "new_lecture.sh"), "9", tpl], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

        lec = os.path.join(root, "Lecture 09")
        out = os.path.join(lec, "Premiere", "Visions of the End L09.prproj")
        self.assertTrue(os.path.isfile(out))
        self.assertEqual(sorted(os.listdir(os.path.join(lec, "Sources"))),
                         sorted(fx.media_name(k, 9) for k in fx.KINDS))
        self.assertEqual(os.listdir(os.path.join(root, "_Raw", "Lecture 9")), [])
        self.assertTrue(os.path.isdir(os.path.join(lec, "Slides")))

        t = tracks(out)
        self.assertEqual([len(x["clips"]) for x in t["V"]], [1, 1, 0, 0, 1, 1])
        self.assertEqual([len(x["clips"]) for x in t["A"]], [1, 0])
        self.assertFalse(any(x["locked"] for x in t["V"] + t["A"]))
        root_xml, ids = load(out)
        names = chain_effects(root_xml, ids)
        self.assertNotIn("AE.ADBE Lumetri", names)
        self.assertNotIn("AE.ADBE Sharpen", names)
        for m in (e for e in root_xml if e.tag == "Media" and e.findtext("Title").endswith(".mp4")):
            for rp in m.findall("RelativePath"):
                self.assertEqual(rp.text, "../Sources/" + m.findtext("Title"))
        text = raw(out)
        self.assertIn(f"{lec}/Premiere/Sequence 01_1.mp4", text)
        self.assertNotIn(tpl_root + "/", text.replace(lec, ""))  # nothing left pointing at the template

    def test_requires_template_argument(self):
        p = subprocess.run(["bash", WRAPPER, "9"], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("template", p.stderr)

    def test_wrong_number_of_recordings(self):
        root = os.path.join(self.tmp, "course2")
        sd = os.path.join(root, "_Scripts")
        os.makedirs(sd)
        for f in ("new_lecture.sh", "make_prproj.py", "prproj.config.json"):
            shutil.copy(os.path.join(SCRIPTS, f), sd)
        fx.make_media(os.path.join(root, "_Raw", "Lecture 3"), 3, kinds=[fx.GAL, fx.SPK])
        p = subprocess.run(["bash", os.path.join(sd, "new_lecture.sh"), "3", self.template],
                           capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("expected 4 .mp4", p.stdout + p.stderr)


if __name__ == "__main__":
    unittest.main()
