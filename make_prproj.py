#!/usr/bin/env python3
"""Clone a Premiere project template and point it at a new set of media files.

usage: make_prproj.py --template T.prproj --media-dir DIR [--sequence NAME] \\
           --keep V1=REGEX [--keep A1=REGEX ...] OUT.prproj
       make_prproj.py --config prproj.config.json [--template ... --media-dir ...] OUT.prproj
Needs ffprobe. Reads the template and media, writes only OUT.prproj.

Config (JSON): any option can be a key, e.g.
  {"sequence": "Sequence 01", "keep": {"V1": " - Gallery View\\\\.mp4$"},
   "strip_effect": ["AE.ADBE Lumetri"], "unlock": true}

Media matching: each template media file is paired with the file in --media-dir
whose name is identical once digit runs are ignored ("Lecture 4 - X.mp4" pairs
with "Lecture 6 - X.mp4"). Clips on template media with no counterpart are
removed from the timeline. .png project items are also removed
from the top level of the project bin.

--keep V<n>=REGEX or A<n>=REGEX (1-based): each track keeps the first clip whose
name matches REGEX, set to start at 0 and run the full length. Tracks without
--keep are left bare. A --keep that matches nothing is an error.

Length: every clip runs the length of the shortest recording. Files that differ by
more than one frame are refused unless --allow-trim is given.

Paths: media paths follow --media-dir. Other paths in the template (export
targets, slide folders) are left alone unless --project-root names the directory
that should replace the template's own; --template-root says which directory
that is when it isn't where the template's media sat.

Orphaned objects are kept: Premiere rejects projects that have them removed.
"""
import argparse, gzip, json, re, subprocess, sys, os
import xml.etree.ElementTree as ET

TICKS_PER_SEC = 254016000000  # Premiere's fixed time base
IMG_EXT = ".png"

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--config", help="JSON file; keys are the option names below (dashes or underscores, "
                "'keep' may be an object like {\"V1\": \"regex\"}). Relative paths resolve against the "
                "config file. Command-line values override it; --keep/--strip-effect entries are added to it")
ap.add_argument("--template", help="existing .prproj to clone")
ap.add_argument("--media-dir", help="directory holding the new media files")
ap.add_argument("--sequence", help="name of the sequence to rebuild (required if the project has several)")
ap.add_argument("--keep", action="append", default=[], metavar="TRACK=REGEX",
                help="e.g. V2=' - Speaker View\\.mp4$'; repeatable")
ap.add_argument("--strip-effect", action="append", default=[], metavar="MATCHNAME",
                help="remove this effect (e.g. 'AE.ADBE Lumetri') from every clip; repeatable")
ap.add_argument("--unlock", action="store_true", help="unlock all tracks")
ap.add_argument("--project-root", help="directory that replaces the template's root directory in non-media "
                "paths (export targets, bin paths). Default: left as in the template")
ap.add_argument("--template-root", help="the directory the template's non-media paths point into "
                "(what --project-root replaces). Default: the directory of the template's media files, "
                "which is only right when the template's media sat directly in its project root")
ap.add_argument("--allow-trim", action="store_true",
                help="allow media of unequal length (everything is cut to the shortest)")
ap.add_argument("out", nargs="?", help="output .prproj (must not exist)")

_pre, _ = ap.parse_known_args()
if _pre.config:
    with open(_pre.config) as f:
        _cfg = json.load(f)
    _cfg_dir = os.path.dirname(os.path.abspath(_pre.config))
    _dests = {a.dest for a in ap._actions} - {"help", "config"}
    _defaults = {}
    for k, v in _cfg.items():
        dest = k.replace("-", "_")
        if dest not in _dests:
            sys.exit(f"unknown key in {_pre.config}: {k!r}")
        if dest in ("template", "media_dir", "project_root", "out") and v and not os.path.isabs(v):
            v = os.path.join(_cfg_dir, v)
        if dest == "keep" and isinstance(v, dict):
            v = [f"{t}={r}" for t, r in v.items()]
        _defaults[dest] = v
    ap.set_defaults(**_defaults)
args = ap.parse_args()
for _req in ("template", "media_dir", "out"):
    if not getattr(args, _req):
        ap.error(f"{_req.replace('_', '-')} is required (command line or --config)")

STAGE = int(os.environ.get("STAGE", "3"))
TEMPLATE = os.path.abspath(args.template)
out = args.out
if not os.path.isfile(TEMPLATE):
    sys.exit(f"template not found: {TEMPLATE}")
if not os.path.isdir(args.media_dir):
    sys.exit(f"media dir not found: {args.media_dir}")
if os.path.exists(out):
    sys.exit(f"refusing to overwrite {out}")
if args.template_root and not args.project_root:
    sys.exit("--template-root has no effect without --project-root")

KEEP_V, KEEP_A = {}, {}
for spec in args.keep:
    label, _, pat = spec.partition("=")
    mm = re.fullmatch(r"([VA])(\d+)", label.strip().upper())
    if not mm or not pat:
        sys.exit(f"--keep expects V<n>=<regex> or A<n>=<regex>, got {spec!r}")
    (KEEP_V if mm.group(1) == "V" else KEEP_A)[int(mm.group(2)) - 1] = re.compile(pat)

def probe(path):
    """Shortest audio/video stream duration in seconds (the container duration is the longest)."""
    try:
        o = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", path],
            text=True, stderr=subprocess.STDOUT)
        d = [float(s["duration"]) for s in json.loads(o)["streams"]
             if s.get("codec_type") in ("audio", "video") and s.get("duration") not in (None, "N/A")]
    except (subprocess.CalledProcessError, ValueError, KeyError) as e:
        sys.exit(f"ffprobe could not read {path}: {e}")
    except FileNotFoundError:
        sys.exit("ffprobe not found; install ffmpeg")
    if not d:
        sys.exit(f"no audio/video stream with a duration in {path}")
    return min(d)

_orig = gzip.open(TEMPLATE).read().decode("utf-8")
_tail = _orig[len(_orig.rstrip()):]
# protect numeric entities (&#10; etc.) so the serializer writes them back verbatim
root = ET.fromstring(re.sub(r"&#(\d+);", "\\1", _orig).encode("utf-8"))

ids = {}
for e in root:
    for k in ("ObjectID", "ObjectUID"):
        if e.get(k):
            ids[e.get(k)] = e

def ref(e):
    return e.get("ObjectRef") or e.get("ObjectURef")

def refs_in(e):
    for c in e.iter():
        r = c.get("ObjectRef") or c.get("ObjectURef")
        if r:
            yield r

def reachable():
    seen = set()
    stack = []
    # roots: top-level elements that are references, like <Project ObjectRef="1"/>
    for e in root:
        if e.get("ObjectRef") or e.get("ObjectURef"):
            stack.append(ref(e))
    while stack:
        k = stack.pop()
        if k in seen or k not in ids:
            continue
        seen.add(k)
        stack.extend(refs_in(ids[k]))
    return seen

before = reachable()

# ---- sequence + its frame duration (ticks/frame) ----
_seqs = [x for x in root if x.tag == "Sequence"]
_all_names = [x.findtext("Name") for x in _seqs]
if args.sequence:
    _seqs = [x for x in _seqs if x.findtext("Name") == args.sequence]
if len(_seqs) != 1:
    sys.exit(f"choose the sequence with --sequence; the project has: {_all_names}")
seq = _seqs[0]
_vgroup = next(g for g in (ids[t.find("Second").get("ObjectRef")] for t in seq.findall(".//TrackGroups/TrackGroup"))
               if g.tag == "VideoTrackGroup")
FRAME = int(_vgroup.find(".//FrameRate").text)

# ---- pair template media with the new files (names equal once digit runs are ignored) ----
def _norm(name):
    return re.sub(r"\d+", "#", name)

_new = {}
for f in sorted(os.listdir(args.media_dir)):
    _new.setdefault(_norm(f), []).append(f)
media_map = {}  # template <Media> element -> new file name
for m in [x for x in root if x.tag == "Media"]:
    cands = _new.get(_norm(m.findtext("Title") or ""), [])
    if len(cands) > 1:
        sys.exit(f"ambiguous: template media {m.findtext('Title')!r} matches {cands}")
    if cands:
        media_map[m] = cands[0]
if not media_map:
    sys.exit(f"no file in {args.media_dir} pairs with any template media")
_taken = {}
for m, nn in media_map.items():
    if nn in _taken:
        sys.exit(f"{nn!r} would replace both {_taken[nn]!r} and {m.findtext('Title')!r}; "
                 f"remove stray files from {args.media_dir}")
    _taken[nn] = m.findtext("Title")
print("media:", *[f"{m.findtext('Title')} -> {nn}" for m, nn in media_map.items()], sep="\n  ")

# durations: every clip runs the length of the shortest file, floored to a frame
secs = [probe(os.path.join(args.media_dir, nn)) for nn in media_map.values()]
if max(secs) - min(secs) > FRAME / TICKS_PER_SEC and not args.allow_trim:
    sys.exit("media lengths differ by more than one frame (s): "
             f"{dict(zip(media_map.values(), secs))}; the longer ones would be cut. Use --allow-trim to proceed")
D = int(min(secs) * (TICKS_PER_SEC / FRAME)) * FRAME  # frames * ticks/frame
print("durations (s):", secs, "-> ticks", D)

if STAGE >= 1:
    def groups(kind):
        out_ = []
        for g in seq.findall(".//TrackGroups/TrackGroup"):
            grp = ids[g.find("Second").get("ObjectRef")]
            if grp.tag == kind:
                out_.append([ids[t.get("ObjectURef")] for t in grp.findall(".//Tracks/Track")])
        return out_

    vtracks = groups("VideoTrackGroup")[0]
    atracks = groups("AudioTrackGroup")[0]

    def item_name(item):
        s = item.find("ClipTrackItem/SubClip")
        if s is None:
            return ""
        return ids[s.get("ObjectRef")].findtext("Name") or ""

    def clip_media(item):
        """The <Media> element a track item plays: SubClip -> Clip -> Source -> MediaSource/Media."""
        sc_ = ids[item.find("ClipTrackItem/SubClip").get("ObjectRef")]
        src = ids[ids[sc_.find("Clip").get("ObjectRef")].find(".//Source").get("ObjectRef")]
        return ids.get(src.find("MediaSource/Media").get("ObjectURef"))

    survivors = set()  # object ids of kept track items

    def process(track, pat):
        clip_ti = track.find(".//ClipItems/TrackItems")
        tr_ti = track.find(".//TransitionItems/TrackItems")
        keep = None
        if clip_ti is not None:
            for ti in list(clip_ti):
                it = ids[ti.get("ObjectRef")]
                if pat is not None and keep is None and pat.search(item_name(it)):
                    keep = ti
            for ti in list(clip_ti):
                if ti is not keep:
                    clip_ti.remove(ti)
        if tr_ti is not None:
            for ti in list(tr_ti):
                tr_ti.remove(ti)
        if keep is None:
            return False
        it = ids[keep.get("ObjectRef")]
        if clip_media(it) not in media_map:
            sys.exit(f"--keep matched clip {item_name(it)!r}, but its media has no counterpart in {args.media_dir}")
        survivors.add(keep.get("ObjectRef"))
        keep.set("Index", "0")
        start = it.find("ClipTrackItem/TrackItem/Start")
        end = it.find("ClipTrackItem/TrackItem/End")
        start.text = "0"
        end.text = str(D)
        for tag in ("HeadTransition", "TailTransition"):
            for x in it.findall(f"ClipTrackItem/{tag}"):
                it.find("ClipTrackItem").remove(x)
        sc = ids[it.find("ClipTrackItem/SubClip").get("ObjectRef")]
        clip = ids[sc.find("Clip").get("ObjectRef")]
        ci = clip.find(".//InPoint")
        co = clip.find(".//OutPoint")
        ci.text = "0"
        co.text = str(D)
        return True

    for label, keep_map, tracks in (("V", KEEP_V, vtracks), ("A", KEEP_A, atracks)):
        for i in keep_map:
            if i >= len(tracks):
                sys.exit(f"--keep {label}{i + 1}: the sequence has only {len(tracks)} {label} tracks")
    _unmatched = []
    for i, t in enumerate(vtracks):
        if not process(t, KEEP_V.get(i)) and i in KEEP_V:
            _unmatched.append(f"V{i + 1}={KEEP_V[i].pattern}")
    for i, t in enumerate(atracks):
        if not process(t, KEEP_A.get(i)) and i in KEEP_A:
            _unmatched.append(f"A{i + 1}={KEEP_A[i].pattern}")
    if _unmatched:
        sys.exit(f"--keep matched no clip: {_unmatched}")
    print("kept items:", len(survivors))

    # ---- links / persistent groups: keep only items that survived ----
    def filt_group(e, parent_of):
        pass

    for pg in [x for x in root if x.tag == "PersistentGroup"]:
        tis = pg.find("TrackItemGroup/TrackItems")
        for t in list(tis):
            if t.get("ObjectRef") not in survivors:
                tis.remove(t)
    for lk in [x for x in root if x.tag == "Link"]:
        tis = lk.find("TrackItemGroup/TrackItems")
        for t in list(tis):
            if t.get("ObjectRef") not in survivors:
                tis.remove(t)
    dead_links = set()
    for lk in [x for x in root if x.tag == "Link"]:
        if len(lk.find("TrackItemGroup/TrackItems")) < 2:
            dead_links.add(lk.get("ObjectID"))
    dead_groups = set(pg.get("ObjectID") for pg in root
                      if pg.tag == "PersistentGroup" and len(pg.find("TrackItemGroup/TrackItems")) < 2)
    if dead_links or dead_groups:
        for e in root:
            for parent in e.iter():
                for c in list(parent):
                    if c.tag in ("Link", "PersistentGroup") and c.get("ObjectRef") in (dead_links | dead_groups):
                        parent.remove(c)
        # drop references to dead groups that sit under other tags
        for e in root:
            for parent in e.iter():
                for c in list(parent):
                    if c.get("ObjectRef") in dead_groups and c.tag != "PersistentGroup":
                        parent.remove(c)
        for e in list(root):
            if e.get("ObjectID") in (dead_links | dead_groups):
                root.remove(e)
                ids.pop(e.get("ObjectID"), None)

    # ---- sequence view state ----
    props = seq.find("Node/Properties")
    def setp(tag, val):
        e = props.find(tag)
        if e is not None:
            e.text = str(val)
    setp("MZ.InPoint", 0)
    setp("MZ.OutPoint", D)
    setp("MZ.EditLine", 0)
    setp("MZ.WorkInPoint", 0)
    setp("MZ.WorkOutPoint", D)
    setp("Monitor.ProgramZoomIn", 0)
    setp("Monitor.ProgramZoomOut", D)
    setp("TL.SQVisibleBaseTime", 0)


if STAGE >= 2:
    # ---- strip template-specific markers (sequence chapters, source-clip markers) ----
    for mk in [x for x in root if x.tag == "Markers"]:
        inner = mk.find("Markers")
        if inner is not None:
            for c in list(inner):
                inner.remove(c)


if STAGE >= 3:
    # ---- project bin: drop .png project items (slides) ----
    rootbin = [x for x in root if x.tag == "RootProjectItem"][0]
    items_el = rootbin.find(".//Items")
    for it in list(items_el):
        t = ids.get(ref(it))
        if (not os.environ.get("SKIPBIN")) and t is not None and t.tag == "ClipProjectItem" and (t.findtext(".//Name") or "").lower().endswith(IMG_EXT):
            items_el.remove(it)

    # ---- delete objects orphaned by the edits (off unless NOGC is set empty; Premiere rejects the result) ----
    after = reachable()
    orphans = set() if os.environ.get("NOGC", "1") else before - after
    for e in list(root):
        k = e.get("ObjectID") or e.get("ObjectUID")
        if k in orphans:
            root.remove(e)
    print("removed orphan objects:", len(orphans))


# ---- retarget media ----
media_dir = os.path.abspath(args.media_dir)
out_dir = os.path.dirname(os.path.abspath(out))
old_durs = set()
path_swap, name_swap, dir_swap = {}, {}, {}  # old file path / old title / old media dir -> new
for m, newname in media_map.items():
    newpath = os.path.join(media_dir, newname)
    oldpath = m.findtext("ActualMediaFilePath") or ""
    path_swap[oldpath] = newpath
    name_swap[m.findtext("Title")] = newname
    if args.project_root:
        old_root = (args.template_root or os.path.dirname(oldpath)).rstrip("/")
        if old_root:
            dir_swap[old_root] = os.path.abspath(args.project_root)
    m.find("Title").text = newname
    for tag in ("ActualMediaFilePath", "FilePath"):
        m.find(tag).text = newpath
    for rp in m.findall("RelativePath"):
        rp.text = os.path.relpath(newpath, out_dir)
    for tag in ("AudioStream", "VideoStream"):
        s = ids[m.find(tag).get("ObjectRef")]
        d = s.find("Duration")
        old_durs.add(d.text)
        d.text = str(D)
    uid = m.get("ObjectUID")
    for e in root:
        if e.tag in ("VideoMediaSource", "AudioMediaSource") and e.find("MediaSource/Media").get("ObjectURef") == uid:
            e.find("OriginalDuration").text = str(D)

for e in root.iter():
    if e.text:
        for swap in (path_swap, name_swap, dir_swap):  # most specific first
            for old, new in swap.items():
                if old and old in e.text:
                    e.text = e.text.replace(old, new)
        if e.text.strip() in old_durs:
            e.text = str(D)

# report elements that still mention template media or .png files
_stale = list(path_swap) + list(name_swap) + list(dir_swap)
left = [e for e in root.iter() if e.text and (any(s in e.text for s in _stale) or e.text.lower().endswith(IMG_EXT))]
print("leftover mentions:", len(left), [e.text[:80] for e in left[:5]])

# ---- strip requested effects from every clip, optionally unlock all tracks ----
_DROP = set(args.strip_effect)
for chain in [x for x in root if x.tag == "VideoComponentChain"]:
    comps = chain.find("ComponentChain/Components")
    if comps is None:
        continue
    for c in list(comps):
        comp = ids.get(c.get("ObjectRef"))
        if comp is not None and comp.findtext(".//MatchName") in _DROP:
            comps.remove(c)
    for i, c in enumerate(comps):
        c.set("Index", str(i))
if args.unlock:
    for parent in root.iter():
        for c in list(parent):
            if c.tag == "IsLocked" and (c.text or "").strip().lower() == "true":
                parent.remove(c)

# ---- re-number Index attributes so sibling lists stay contiguous ----
for p in root.iter():
    kids = [c for c in p if c.get("Index") is not None]
    for i, c in enumerate(kids):
        c.set("Index", str(i))

xml = ET.tostring(root, encoding="unicode")
xml = re.sub("(\\d+)", r"&#\1;", xml)
xml = xml.replace(" />", "/>")  # match Premiere's own self-closing style
xml = '<?xml version="1.0" encoding="UTF-8" ?>\n' + xml + _tail
with gzip.open(out, "wb") as f:
    f.write(xml.encode("utf-8"))
print("wrote", out)
