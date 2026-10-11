#!/usr/bin/env python3
"""Remove Lumetri Color + Sharpen from every clip and unlock all tracks.

usage: strip_color_unlock.py <project.prproj> [more.prproj ...]
Rewrites each file in place. Each original is first copied to <name>.prproj.bak
next to it, or into the directory named by the BACKUP_DIR environment variable.
"""
import gzip, os, re, shutil, sys
import xml.etree.ElementTree as ET

DROP = {"AE.ADBE Lumetri", "AE.ADBE Sharpen"}


def process(path):
    backup_dir = os.environ.get("BACKUP_DIR")
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))
    else:
        shutil.copy2(path, path + ".bak")

    orig = gzip.open(path).read().decode("utf-8")
    tail = orig[len(orig.rstrip()):]
    root = ET.fromstring(re.sub(r"&#(\d+);", "\\1", orig).encode("utf-8"))
    ids = {}
    for e in root:
        for k in ("ObjectID", "ObjectUID"):
            if e.get(k):
                ids[e.get(k)] = e

    removed = 0
    for chain in [x for x in root if x.tag == "VideoComponentChain"]:
        comps = chain.find("ComponentChain/Components")
        if comps is None:
            continue
        for c in list(comps):
            comp = ids.get(c.get("ObjectRef"))
            if comp is not None and comp.findtext(".//MatchName") in DROP:
                comps.remove(c)
                removed += 1
        for i, c in enumerate(comps):
            c.set("Index", str(i))
        # keep the active-component pointer valid
        act = chain.find("ComponentChain/Node/Properties/MZ.ComponentChain.ActiveComponentID")
        if act is not None:
            live = set()
            for c in comps:
                comp = ids.get(c.get("ObjectRef"))
                cid = comp.findtext(".//ComponentID") if comp is not None else None
                if cid:
                    live.add(cid)
            if live and act.text not in live and act.text != "4294967295":
                dflt = chain.findtext("DefaultOpacityComponentID")
                act.text = dflt if dflt in live else sorted(live)[0]
            act = chain.find("ComponentChain/Node/Properties/MZ.ComponentChain.ActiveComponentParamIndex")
            if act is not None:
                act.text = "4294967295"

    unlocked = 0
    for parent in root.iter():
        for c in list(parent):
            if c.tag == "IsLocked" and (c.text or "").strip().lower() == "true":
                parent.remove(c)
                unlocked += 1

    xml = ET.tostring(root, encoding="unicode")
    xml = re.sub("(\\d+)", r"&#\1;", xml).replace(" />", "/>")
    out = '<?xml version="1.0" encoding="UTF-8" ?>\n' + xml + tail
    with gzip.open(path, "wb") as f:
        f.write(out.encode("utf-8"))
    print(f"{os.path.basename(path)}: removed {removed} effect entries, unlocked {unlocked} tracks")


for p in sys.argv[1:]:
    process(p)
