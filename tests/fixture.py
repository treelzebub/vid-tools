"""Synthetic Premiere project + media for testing make_prproj.py.

The project mirrors the object chain the script relies on (Sequence -> TrackGroups ->
tracks -> track items -> SubClip -> Clip -> Source -> Media -> streams), laid out like
the real lecture template: six video tracks, two audio tracks, slide stills on V4,
a clip on V3 and A2 that the config deliberately empties, effects, markers, links,
locked tracks, transitions, a second (empty) sequence, and `&#10;` entities.
"""
import gzip
import itertools
import os
import subprocess

TICKS_PER_SEC = 254016000000
FPS25 = 10160640000  # ticks per frame at 25 fps
FPS30 = 8467200000  # ticks per frame at 30 fps
PREFIX = "Visions of the End"
KINDS = ["Gallery View", "Speaker View", "Shared Screen", "Shared Screen with Speaker View"]
GAL, SPK, SS, SSWSV = KINDS
SLIDES = ["slide-01.png", "slide-02.png"]

# (track label, [clip names left to right])
VIDEO_TRACKS = [
    ("V1", [GAL, GAL]),
    ("V2", [GAL, SPK, SPK, SPK]),
    ("V3", [SPK, SPK]),
    ("V4", SLIDES),
    ("V5", [SSWSV, SSWSV]),
    ("V6", [SS, SS]),
]
AUDIO_TRACKS = [("A1", [SSWSV, SSWSV]), ("A2", [SSWSV])]


def media_name(kind, n):
    return f"{PREFIX} - Lecture {n} - {kind}.mp4"


def make_media(directory, n, seconds=2, kinds=KINDS):
    """Write tiny real mp4s (25 fps, with audio) named for lecture n."""
    os.makedirs(directory, exist_ok=True)
    paths = {}
    for kind in kinds:
        p = os.path.join(directory, media_name(kind, n))
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y",
             "-f", "lavfi", "-i", f"testsrc=size=160x90:rate=25:duration={seconds}",
             "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", p],
            check=True)
        paths[kind] = p
    return paths


class _Ids:
    def __init__(self):
        self._n = itertools.count(100)
        self._u = itertools.count(1)

    def num(self):
        return str(next(self._n))

    def uid(self):
        return f"00000000-0000-4000-8000-{next(self._u):012d}"


def build(path, root_dir, n=4, frame=FPS25, duration_s=3600, effects=True, locked=True):
    """Write a gzip .prproj at `path`. `root_dir` is the directory that its media and
    export paths point into (the template's project root)."""
    ids = _Ids()
    objs = []
    dur = duration_s * TICKS_PER_SEC

    # ---- media ----
    media_uid, source_id = {}, {}
    for kind in KINDS:
        name = media_name(kind, n)
        full = f"{root_dir}/{name}"
        uid, vs, as_, vms, ams = ids.uid(), ids.num(), ids.num(), ids.num(), ids.num()
        media_uid[kind], source_id[kind] = uid, (vms, ams)
        objs.append(
            f'<Media ObjectUID="{uid}" ClassID="7a5c103e" Version="30">'
            f'<AudioStream ObjectRef="{as_}"/><VideoStream ObjectRef="{vs}"/>'
            f'<RelativePath>../{PREFIX}/{name}</RelativePath>'
            f'<RelativePath>../{name}</RelativePath>'
            f'<ActualMediaFilePath>{full}</ActualMediaFilePath><FilePath>{full}</FilePath>'
            f'<Title>{name}</Title></Media>')
        objs.append(f'<VideoStream ObjectID="{vs}" ClassID="a"><Duration>{dur}</Duration>'
                    f'<ConformedAudioPath>{root_dir}/Conformed/{name}.cfa</ConformedAudioPath>'
                    f'<PeakFilePath>{root_dir}/Peaks/{name}.pek</PeakFilePath></VideoStream>')
        objs.append(f'<AudioStream ObjectID="{as_}" ClassID="b"><Duration>{dur}</Duration></AudioStream>')
        objs.append(f'<VideoMediaSource ObjectID="{vms}" ClassID="c"><MediaSource Version="4">'
                    f'<Media ObjectURef="{uid}"/></MediaSource>'
                    f'<OriginalDuration>{dur}</OriginalDuration></VideoMediaSource>')
        objs.append(f'<AudioMediaSource ObjectID="{ams}" ClassID="d"><MediaSource Version="4">'
                    f'<Media ObjectURef="{uid}"/></MediaSource>'
                    f'<OriginalDuration>{dur}</OriginalDuration></AudioMediaSource>')
    for sname in SLIDES:  # still-image media, no counterpart in any new media dir
        uid, vs, as_, vms = ids.uid(), ids.num(), ids.num(), ids.num()
        media_uid[sname], source_id[sname] = uid, (vms, None)
        full = f"{root_dir}/Slides/{sname}"
        objs.append(
            f'<Media ObjectUID="{uid}" ClassID="7a5c103e" Version="30">'
            f'<AudioStream ObjectRef="{as_}"/><VideoStream ObjectRef="{vs}"/>'
            f'<RelativePath>../Slides/{sname}</RelativePath>'
            f'<ActualMediaFilePath>{full}</ActualMediaFilePath><FilePath>{full}</FilePath>'
            f'<Title>{sname}</Title></Media>')
        objs.append(f'<VideoStream ObjectID="{vs}" ClassID="a"><Duration>{5 * TICKS_PER_SEC}</Duration></VideoStream>')
        objs.append(f'<AudioStream ObjectID="{as_}" ClassID="b"><Duration>0</Duration></AudioStream>')
        objs.append(f'<VideoMediaSource ObjectID="{vms}" ClassID="c"><MediaSource Version="4">'
                    f'<Media ObjectURef="{uid}"/></MediaSource>'
                    f'<OriginalDuration>{5 * TICKS_PER_SEC}</OriginalDuration></VideoMediaSource>')

    # ---- tracks ----
    item_ids = {}  # (label, index) -> track item id
    track_uids = {}

    def comp_chain(with_effects):
        comps = []
        defs = [("AE.ADBE Motion", "Motion"), ("AE.ADBE Opacity", "Opacity")]
        if with_effects:
            defs += [("AE.ADBE Lumetri", "Lumetri Color"), ("AE.ADBE Sharpen", "Sharpen")]
        for mn, dn in defs:
            cid = ids.num()
            objs.append(f'<VideoFilterComponent ObjectID="{cid}" ClassID="e"><Component Version="7">'
                        f'<MatchName>{mn}</MatchName><DisplayName>{dn}</DisplayName>'
                        f'<ComponentID>{cid}</ComponentID></Component></VideoFilterComponent>')
            comps.append(cid)
        chain = ids.num()
        body = "".join(f'<Component Index="{i}" ObjectRef="{c}"/>' for i, c in enumerate(comps))
        objs.append(f'<VideoComponentChain ObjectID="{chain}" ClassID="f"><ComponentChain Version="3">'
                    f'<Node Version="1"><Properties Version="1">'
                    f'<MZ.ComponentChain.ActiveComponentID>{comps[0]}</MZ.ComponentChain.ActiveComponentID>'
                    f'</Properties></Node><Components Version="1">{body}</Components></ComponentChain>'
                    f'</VideoComponentChain>')
        return chain

    def track(label, clips, video):
        uid = ids.uid()
        track_uids[label] = uid
        items, trans = [], []
        for i, nm in enumerate(clips):
            media_key = nm if nm in media_uid else nm
            vms, ams = source_id[media_key]
            tiid, scid, clipid, mkid = ids.num(), ids.num(), ids.num(), ids.num()
            src = vms if video else ams
            start = i * 100 * TICKS_PER_SEC
            title = nm if nm in SLIDES else media_name(nm, n)
            objs.append(f'<Markers ObjectID="{mkid}" ClassID="m"><Markers Version="1">'
                        f'<Marker Index="0" ObjectRef="{ids.num()}"/></Markers></Markers>')
            tag = "VideoClip" if video else "AudioClip"
            objs.append(f'<{tag} ObjectID="{clipid}" ClassID="g"><Clip Version="18">'
                        f'<MarkerOwner Version="1"><Markers ObjectRef="{mkid}"/></MarkerOwner>'
                        f'<Source ObjectRef="{src}"/>'
                        f'<InPoint>{start + 7}</InPoint><OutPoint>{start + 50 * TICKS_PER_SEC}</OutPoint>'
                        f'</Clip></{tag}>')
            objs.append(f'<SubClip ObjectID="{scid}" ClassID="h"><Clip ObjectRef="{clipid}"/>'
                        f'<MasterClip ObjectURef="{ids.uid()}"/><Name>{title}</Name></SubClip>')
            head = tail = ""
            if video and label == "V1":
                h, t = ids.num(), ids.num()
                objs.append(f'<VideoTransitionTrackItem ObjectID="{h}" ClassID="t"/>')
                objs.append(f'<VideoTransitionTrackItem ObjectID="{t}" ClassID="t"/>')
                head, tail = f'<HeadTransition ObjectRef="{h}"/>', f'<TailTransition ObjectRef="{t}"/>'
                trans += [h, t]
            owner = comp_chain(effects) if video else ""
            ctag = "VideoClipTrackItem" if video else "AudioClipTrackItem"
            objs.append(
                f'<{ctag} ObjectID="{tiid}" ClassID="i"><ClipTrackItem Version="8">'
                f'<TrackItem Version="4"><Start>{start}</Start><End>{start + 50 * TICKS_PER_SEC}</End></TrackItem>'
                f'<SubClip ObjectRef="{scid}"/>{head}{tail}</ClipTrackItem></{ctag}>')
            items.append(tiid)
            item_ids[(label, i)] = tiid
        lock = "<IsLocked>true</IsLocked>" if locked else ""
        ttag = "VideoClipTrack" if video else "AudioClipTrack"
        ci = "".join(f'<TrackItem Index="{i}" ObjectRef="{t}"/>' for i, t in enumerate(items))
        ti = "".join(f'<TrackItem Index="{i}" ObjectRef="{t}"/>' for i, t in enumerate(trans))
        objs.append(
            f'<{ttag} ObjectUID="{uid}" ClassID="j"><ClipTrack Version="2"><Track Version="4">'
            f'<Node Version="1"><Properties Version="1"><MZ.TrackName>{label}</MZ.TrackName></Properties></Node>'
            f'<ID>1</ID><Index>0</Index>{lock}</Track>'
            f'<ClipItems Version="3"><TrackItems Version="1">{ci}</TrackItems></ClipItems>'
            f'<TransitionItems Version="3"><TrackItems Version="1">{ti}</TrackItems></TransitionItems>'
            f'</ClipTrack></{ttag}>')
        return uid

    vuids = [track(lbl, clips, True) for lbl, clips in VIDEO_TRACKS]
    auids = [track(lbl, clips, False) for lbl, clips in AUDIO_TRACKS]

    def group(tag, uids, with_rate):
        gid = ids.num()
        refs = "".join(f'<Track Index="{i}" ObjectURef="{u}"/>' for i, u in enumerate(uids))
        rate = f"<FrameRate>{frame}</FrameRate>" if with_rate else ""
        objs.append(f'<{tag} ObjectID="{gid}" ClassID="k"><TrackGroup Version="1">'
                    f'<Tracks Version="1">{refs}</Tracks>{rate}</TrackGroup></{tag}>')
        return gid

    vg, ag = group("VideoTrackGroup", vuids, True), group("AudioTrackGroup", auids, False)
    empty_vg, empty_ag = group("VideoTrackGroup", [], True), group("AudioTrackGroup", [], False)

    def sequence(name, v, a, uid=None):
        return (
            f'<Sequence ObjectUID="{uid or ids.uid()}" ClassID="s" Version="12"><Node Version="1">'
            f'<Properties Version="1"><MZ.InPoint>111</MZ.InPoint><MZ.OutPoint>222</MZ.OutPoint>'
            f'<MZ.EditLine>333</MZ.EditLine><MZ.WorkInPoint>444</MZ.WorkInPoint>'
            f'<MZ.WorkOutPoint>555</MZ.WorkOutPoint><Monitor.ProgramZoomIn>666</Monitor.ProgramZoomIn>'
            f'<Monitor.ProgramZoomOut>777</Monitor.ProgramZoomOut><TL.SQVisibleBaseTime>888</TL.SQVisibleBaseTime>'
            f'<ExportSettings.Info.SaveAsFile>{{"OutPath":"{root_dir}/Premiere/{name}_1.mp4","WorkOut":999}}'
            f'</ExportSettings.Info.SaveAsFile></Properties></Node>'
            f'<TrackGroups Version="1"><TrackGroup Index="0"><First>0</First><Second ObjectRef="{v}"/></TrackGroup>'
            f'<TrackGroup Index="1"><First>1</First><Second ObjectRef="{a}"/></TrackGroup></TrackGroups>'
            f'<Name>{name}</Name></Sequence>')

    objs.append(sequence("1080p 23.98", empty_vg, empty_ag))
    objs.append(sequence("Sequence 01", vg, ag))

    # ---- links / persistent groups ----
    def grp(tag, members):
        refs = "".join(f'<TrackItem Index="{i}" ObjectRef="{m}"/>' for i, m in enumerate(members))
        return (f'<{tag} ObjectID="{ids.num()}" ClassID="l"><TrackItemGroup Version="1">'
                f'<TrackItems Version="1">{refs}</TrackItems></TrackItemGroup></{tag}>')
    # first link: both members survive. second link: a slide member is removed, leaving one
    objs.append(grp("Link", [item_ids[("V5", 0)], item_ids[("A1", 0)]]))
    objs.append(grp("Link", [item_ids[("V6", 0)], item_ids[("V4", 0)]]))
    objs.append(grp("PersistentGroup", [item_ids[("V4", 1)], item_ids[("V1", 0)], item_ids[("V2", 1)]]))

    # ---- project bin ----
    bin_items = []
    # stills are interleaved with the videos, as in a real bin, so removing them leaves Index gaps
    for nm in [SLIDES[0], KINDS[0], KINDS[1], SLIDES[1], KINDS[2], KINDS[3]]:
        pid = ids.num()
        shown = nm if nm in SLIDES else media_name(nm, n)
        objs.append(f'<ClipProjectItem ObjectID="{pid}" ClassID="p"><ProjectItem Version="1">'
                    f'<Name>{shown}</Name></ProjectItem></ClipProjectItem>')
        bin_items.append(pid)
    items = "".join(f'<Item Index="{i}" ObjectRef="{b}"/>' for i, b in enumerate(bin_items))
    objs.append(f'<RootProjectItem ObjectID="{ids.num()}" ClassID="q"><ProjectItemContainer Version="1">'
                f'<Items Version="1">{items}</Items></ProjectItemContainer></RootProjectItem>')

    project = (
        '<Project ObjectID="1" ClassID="62ad66dd" Version="45"><Node Version="1"><Properties Version="1">'
        '<Project.Schema>&lt;?xml version="1.0"?&gt;&#10;&lt;defs&gt;&#10;&#9;&lt;x/&gt;&#10;&lt;/defs&gt;</Project.Schema>'
        f'<lastknowngoodprojectpath>{root_dir}/Premiere/template.prproj</lastknowngoodprojectpath>'
        '</Properties></Node></Project>')
    xml = ('<?xml version="1.0" encoding="UTF-8" ?>\n<PremiereData Version="3">\n'
           '\t<Project ObjectRef="1"/>\n\t' + project + "\n\t" + "\n\t".join(objs) + "\n</PremiereData>\n")
    with gzip.open(path, "wb") as f:
        f.write(xml.encode("utf-8"))
