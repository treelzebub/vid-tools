#!/usr/bin/env python3
"""Build .cube LUTs that pull a target video's color toward a reference look.

usage: match_look.py --reference REF TARGET OUT_DIR
  REF     a graded video or a single graded .png frame of the same person
  TARGET  the video to correct
  OUT_DIR where the LUTs, preview, and report are written

Detects the face in sampled frames of the reference and of the target, and measures
skin tone in each (skin-colored pixels of the cheek band, outlier frames dropped).

The correction is applied in linear light as per-channel gains: an exposure gain
(skin brightness moves --skin-pull of the way to the reference) and a white-balance
gain (skin chroma moves --chroma-pull of the way, each channel capped at --max-wb),
followed by a highlight shoulder and a black point. Colors that aren't skin are
not curved individually, so shirts and walls keep their relationships.

Writes, named after --name (default: the target's file name):
  <name> match.cube, <name> match mild.cube, <name> match neutral.cube
      full correction; 40% of the white balance; exposure only
  <name> match-preview.png   target frames: raw | neutral | mild | full | reference
  <name> match-report.json   measurements, warnings, approximate Lumetri equivalents
Load a .cube in Lumetri Color > Creative > Look > Browse, or Basic > Input LUT.

Needs ffmpeg/ffprobe, numpy, opencv-python, and Pillow.
"""
import argparse, json, os, subprocess, sys
import io
import numpy as np
import cv2
from PIL import Image

W = np.array([0.2126, 0.7152, 0.0722])  # Rec.709 luma weights
VARIANTS = {"neutral": 0.0, "mild": 0.4, "full": 1.0}  # share of the white balance applied
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def is_still(path):
    return path.lower().endswith(".png")


def dur(p):
    return float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], text=True))


def frame(path, sec=0.0):
    """One RGB frame (uint8 array) at `sec`, or the image itself for a .png."""
    if is_still(path):
        return np.asarray(Image.open(path).convert("RGB"))
    png = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(sec), "-i", path, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        check=True, capture_output=True).stdout
    if not png:  # seeking past the end of the file
        return None
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def measure(img):
    """-> median skin RGB of the face's cheek band, or None if there is no single clear face."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(g, 1.1, 6, minSize=(120, 120))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # cheeks + forehead: central band, skip eyes/mouth/glasses rows
    patch = np.ascontiguousarray(img[y + int(h * 0.55):y + int(h * 0.80), x + int(w * 0.25):x + int(w * 0.75)])
    ycc = cv2.cvtColor(patch, cv2.COLOR_RGB2YCrCb).reshape(-1, 3)
    px = patch.reshape(-1, 3)
    # keep skin-colored pixels only; this drops glasses frames, hair, shadows, and background
    keep = (ycc[:, 1] >= 133) & (ycc[:, 1] <= 173) & (ycc[:, 2] >= 77) & (ycc[:, 2] <= 127)
    if keep.mean() < 0.4:
        return None
    m = np.median(px[keep], 0)
    if not (m[0] > m[1] > m[2] * 0.9 and m[0] > 40):
        return None
    return m


def collect(path, times, min_frames=3):
    """Median skin color over the usable frames, with counts and brightness spread for the report.

    `usable_times` lists the sample times whose frame had a clear face and survived the outlier test.
    """
    found = []  # (time, skin rgb)
    for t in times:
        img = frame(path, t)
        s = measure(img) if img is not None else None
        if s is not None:
            found.append((t, s))
    if len(found) < min_frames:
        sys.exit(f"too few usable frames from {path} ({len(found)} of {len(times)} had a clear face)")
    skins = np.array([s for _, s in found])
    # drop frames whose skin brightness is far from the rest (hand over the face, glare, head turn)
    y = skins @ W
    mad = max(float(np.median(np.abs(y - np.median(y)))), 1.0)
    good = np.abs(y - np.median(y)) <= 3 * 1.4826 * mad
    kept = skins[good]
    spread = float(np.std(np.log2(to_lin(kept) @ W)))
    return dict(skin=np.median(kept, 0), tried=len(times), found=len(found), used=int(good.sum()),
                spread_stops=spread, usable_times=[t for (t, _), g in zip(found, good) if g])


def to_lin(c):
    """sRGB 0-255 -> linear 0-1 (treats Rec.709 video as sRGB, which is close enough here)."""
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(lin):
    lin = np.clip(lin, 0.0, 1.0)
    return 255.0 * np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def lab(srgb):
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ to_lin(srgb)
    xyz = xyz / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e(a, b):
    """CIE76 colour difference between two sRGB colors (0-255)."""
    return float(np.linalg.norm(lab(a) - lab(b)))


class Correction:
    """Gains computed in linear light from a skin measurement of the target and of the reference.

    Exposure moves skin brightness only part of the way, so the room isn't darkened to match
    (a reference's shelves can be dark because that footage was dark, not because of the grade).
    """

    def __init__(self, target_skin, ref_skin, skin_pull=0.6, chroma_pull=0.75, max_wb=1.3,
                 black_point=10.0, highlight_cap=250.0):
        self.black_point, self.highlight_cap, self.max_wb = black_point, highlight_cap, max_wb
        self.lin = to_lin(target_skin)
        ref_lin = to_lin(ref_skin)
        self.y = float(self.lin @ W)
        ref_y = float(ref_lin @ W)
        self.exposure = (ref_y / self.y) ** skin_pull  # one gain on all channels
        tgt_lin = ref_lin * (self.y * self.exposure / ref_y)  # reference chroma at the new brightness
        wb = np.log(tgt_lin / self.lin / self.exposure) * chroma_pull  # per-channel white balance, log gains
        self.wb_clamped = bool(np.any(np.abs(wb) > np.log(max_wb)))
        self.wb_log = np.clip(wb, -np.log(max_wb), np.log(max_wb))

    @property
    def stops(self):
        return float(np.log2(self.exposure))

    def wb_gains(self, alpha):
        """White-balance gains at strength alpha (0 = none), renormalized so skin keeps its brightness."""
        g = np.exp(alpha * self.wb_log)
        return g * (self.y / float((self.lin * g) @ W))

    def transform(self, rgb, alpha):
        """rgb: (...,3) 0-255 -> corrected 0-255. alpha scales the white balance: 0 = exposure only, 1 = full."""
        gain = self.exposure * self.wb_gains(alpha)
        lin = to_lin(rgb) * gain
        # highlight shoulder: a channel that is being brightened rolls off above 0.8 instead of clipping,
        # so brightening doesn't blow out the shirt; a channel that isn't is left alone
        knee = np.where(gain > 1.0, 0.8, 1.0)
        room = 1.0 - knee
        over = np.maximum(lin - knee, 0.0)
        lin = np.minimum(lin, knee) + np.where(room > 0, room * np.tanh(over / np.where(room > 0, room, 1.0)), 0.0)
        out = to_srgb(lin)
        # black point and highlight cap, in display values
        b = self.black_point
        return np.clip((out - b) / (255.0 - b), 0.0, 1.0) * self.highlight_cap


def write_lut(path, corr, alpha, title, size=33):
    grid = np.linspace(0, 255, size)
    b, g, r = np.meshgrid(grid, grid, grid, indexing="ij")  # R varies fastest in .cube
    cube = np.stack([r, g, b], -1).reshape(-1, 3)
    with open(path, "w") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {size}\n')
        np.savetxt(f, corr.transform(cube, alpha) / 255.0, fmt="%.6f")


def write_preview(path, corr, target, target_dur, ref):
    """One row per pose (early, middle, late); columns are raw, neutral, mild, full, reference."""
    ref_times = ref["usable_times"] or [0.0]
    rows = []
    for frac in (0.15, 0.50, 0.85):
        src = frame(target, target_dur * frac)
        if src is None:
            continue
        src = src.astype(float)
        ref_img = frame(ref["path"], ref_times[min(int(frac * len(ref_times)), len(ref_times) - 1)])
        if ref_img is None:
            ref_img = src
        elif ref_img.shape != src.shape:  # e.g. a still exported at another size
            ref_img = np.asarray(Image.fromarray(ref_img).resize(src.shape[1::-1]))
        rows.append(np.concatenate([src] + [corr.transform(src, a) for a in VARIANTS.values()]
                                   + [ref_img.astype(float)], 1))
    if rows:
        prev = np.concatenate(rows, 0).astype(np.uint8)
        Image.fromarray(prev).resize((prev.shape[1] // 3, prev.shape[0] // 3)).save(path)


def build_report(corr, target, ref, target_skin, ref_skin, name):
    """What was measured, how far the result is from the reference, and what to distrust."""
    stops, wb = corr.stops, np.exp(corr.wb_log)
    warnings = []
    for label, m in (("reference", ref), ("target", target)):
        if m["used"] < 8 and m["tried"] > 1:
            warnings.append(f"only {m['used']} of {m['tried']} frames of the {label} gave a usable face; "
                            "the skin estimate is weak")
        if m["spread_stops"] > 0.3:
            warnings.append(f"skin brightness in the {label} varies by {m['spread_stops']:.2f} stops (std); "
                            "auto exposure drifts, so one correction will not fit the whole video")
    if corr.wb_clamped:
        warnings.append(f"the white-balance gap exceeds the {corr.max_wb}x cap on at least one channel; "
                        "check the source footage")
    if abs(stops) > 0.7:
        warnings.append(f"large exposure change ({stops:+.2f} stops); the source may be under- or overexposed")
    after = {k: corr.transform(target_skin, a) for k, a in VARIANTS.items()}
    keys = ("tried", "found", "used", "spread_stops")
    return {
        "name": name,
        "reference_frames": {k: ref[k] for k in keys},
        "target_frames": {k: target[k] for k in keys},
        "skin_rgb": {"raw": target_skin.round(1).tolist(), "reference": ref_skin.round(1).tolist(),
                     **{k: v.round(1).tolist() for k, v in after.items()}},
        "skin_delta_e76_to_reference": {"raw": round(delta_e(target_skin, ref_skin), 1),
                                        **{k: round(delta_e(v, ref_skin), 1) for k, v in after.items()}},
        "gains": {"exposure_stops": round(stops, 2), "white_balance_rgb": wb.round(3).tolist(),
                  "black_point_0_255": corr.black_point, "highlight_cap_0_255": corr.highlight_cap},
        "lumetri_approx": {
            "note": "direction is reliable, magnitude needs tuning by eye; the LUT is the exact version",
            "exposure_stops": round(stops, 2),
            "temperature": "warmer" if wb[0] > wb[2] else "cooler",
            "red_to_blue_gain_ratio": round(float(wb[0] / wb[2]), 3),
            "tint": "toward green" if wb[1] > np.sqrt(wb[0] * wb[2]) else "toward magenta",
            "green_to_red_blue_gain_ratio": round(float(wb[1] / np.sqrt(wb[0] * wb[2])), 3),
        },
        "warnings": warnings,
    }


def measure_reference(path, times, samples, cache):
    """Measure the reference once; with a cache file, reuse the result while the file is unchanged."""
    if is_still(path):
        r = collect(path, [0.0], min_frames=1)
        return dict(r, path=path)
    d = dur(path)
    if not times:
        times = [float(t) for t in np.linspace(0.05, 0.95, samples) * d]
    key = {"path": os.path.abspath(path), "mtime": os.path.getmtime(path), "size": os.path.getsize(path),
           "times": [round(t, 3) for t in times], "version": 1}
    if cache and os.path.isfile(cache):
        try:
            with open(cache) as f:
                hit = json.load(f)
            if hit.get("key") == key:
                r = hit["result"]
                r["skin"] = np.array(r["skin"])
                return dict(r, path=path, duration=d)
        except (ValueError, KeyError, OSError):
            pass
    r = collect(path, times)
    if cache:
        with open(cache, "w") as f:
            json.dump({"key": key, "result": {**r, "skin": r["skin"].tolist()}}, f)
    return dict(r, path=path, duration=d)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", required=True, help="graded video, or a single graded .png frame")
    ap.add_argument("target", help="video to correct")
    ap.add_argument("out", help="output directory (created if missing)")
    ap.add_argument("--name", help="prefix for the output files (default: the target's file name)")
    ap.add_argument("--reference-times", type=lambda s: [float(x) for x in s.split(",")],
                    help="comma-separated seconds in the reference to sample (default: spread across it)")
    ap.add_argument("--reference-samples", type=int, default=40, help="reference frames to try (default 40)")
    ap.add_argument("--target-samples", type=int, default=16, help="target frames to try (default 16)")
    ap.add_argument("--cache", help="JSON file that stores the reference measurement between runs")
    ap.add_argument("--skin-pull", type=float, default=0.6, help="share of the exposure gap closed (0.6)")
    ap.add_argument("--chroma-pull", type=float, default=0.75, help="share of the white-balance gap closed (0.75)")
    ap.add_argument("--max-wb", type=float, default=1.3, help="cap on any one channel's white-balance gain (1.3)")
    ap.add_argument("--black-point", type=float, default=10.0, help="display value 0-255 mapped to black (10)")
    ap.add_argument("--highlight-cap", type=float, default=250.0, help="display value 0-255 for white (250)")
    ap.add_argument("--lut-size", type=int, default=33, help="points per axis of the 3D LUT (33)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    for label, p in (("reference", a.reference), ("target", a.target)):
        if not os.path.isfile(p):
            sys.exit(f"{label} not found: {p}")
    if is_still(a.target):
        sys.exit("the target must be a video")
    os.makedirs(a.out, exist_ok=True)
    name = a.name or os.path.splitext(os.path.basename(a.target))[0]

    ref = measure_reference(a.reference, a.reference_times, a.reference_samples, a.cache)
    d = dur(a.target)
    target = collect(a.target, [float(t) for t in np.linspace(0.08, 0.92, a.target_samples) * d])
    corr = Correction(target["skin"], ref["skin"], a.skin_pull, a.chroma_pull, a.max_wb,
                      a.black_point, a.highlight_cap)

    for variant, alpha in VARIANTS.items():
        suffix = "" if variant == "full" else f" {variant}"
        path = os.path.join(a.out, f"{name} match{suffix}.cube")
        write_lut(path, corr, alpha, f"{name} match ({variant})", a.lut_size)
        print("wrote", path)
    write_preview(os.path.join(a.out, f"{name} match-preview.png"), corr, a.target, d, ref)

    report = build_report(corr, target, ref, target["skin"], ref["skin"], name)
    with open(os.path.join(a.out, f"{name} match-report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"frames  reference {ref['used']}/{ref['tried']}  target {target['used']}/{target['tried']}")
    print("skin    target", target["skin"].round(0), " reference", ref["skin"].round(0))
    print("dE76 to reference:", report["skin_delta_e76_to_reference"])
    print(f"exposure {corr.stops:+.2f} stops   white balance R,G,B x{report['gains']['white_balance_rgb']}")
    for w in report["warnings"]:
        print("WARNING:", w)


if __name__ == "__main__":
    main()
