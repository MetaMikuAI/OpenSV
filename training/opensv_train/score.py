import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


QUARTER = 705600000


def text(node, name, default=None):
    child = node.find(name)
    if child is None or child.text is None:
        if default is None:
            raise ValueError(f"VSQX 缺少 {name}")
        return default
    return child.text


def vsqx(path):
    root = ET.fromstring(path.read_bytes())
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    if root.tag not in ("vsq3", "vsq4"):
        raise ValueError("只支持 VSQX 3／4 格式")
    v4 = root.tag == "vsq4"
    tick, bpm, measure, num, den = ("t", "v", "m", "nu", "de") if v4 else (
        "posTick", "bpm", "posMes", "nume", "denomi")
    master = root.find("masterTrack")
    if master is None:
        raise ValueError("VSQX 缺少 masterTrack")
    resolution = int(text(master, "resolution"))
    pre = int(text(master, "preMeasure", "0"))
    if resolution <= 0 or not 0 <= pre <= 100:
        raise ValueError("VSQX 拍长或预备小节无效")
    signatures = sorted((int(text(x, measure)), int(text(x, num)), int(text(x, den)))
                        for x in master.findall("timeSig"))
    if not signatures or signatures[0][0] != 0 or any(n <= 0 or d <= 0 for _, n, d in signatures):
        raise ValueError("VSQX 拍号无效")
    pre_ticks = 0
    for bar in range(pre):
        _, n, d = next(s for s in reversed(signatures) if s[0] <= bar)
        pre_ticks += resolution * 4 * n / d
    tempos = sorted((int(text(x, tick)), float(text(x, bpm)) / 100) for x in master.findall("tempo"))
    if not tempos or tempos[0][0] != 0:
        raise ValueError("VSQX 缺少起始速度")
    initial = next(t[1] for t in reversed(tempos) if t[0] <= pre_ticks)
    tempo = [dict(position=0, bpm=initial)] + [dict(position=round((t-pre_ticks)*QUARTER/resolution), bpm=b)
                                             for t, b in tempos if t > pre_ticks]
    tracks = []
    for track in root.findall("vsTrack"):
        notes = []
        for part in track.findall("vsPart" if v4 else "musicalPart"):
            origin = int(text(part, tick)) - pre_ticks
            for note in part.findall("note"):
                notes.append(dict(onset=round((origin+int(text(note, tick)))*QUARTER/resolution),
                                  duration=round(int(text(note, "dur" if v4 else "durTick"))*QUARTER/resolution),
                                  pitch=int(text(note, "n" if v4 else "noteNum")),
                                  lyrics=text(note, "y" if v4 else "lyric")))
        if notes:
            tracks.append(dict(mainGroup=dict(notes=notes)))
    return dict(time=dict(tempo=tempo), tracks=tracks)


def load_score(path):
    path = Path(path)
    if path.suffix.lower() == ".vsqx":
        project = vsqx(path)
    elif path.suffix.lower() == ".svp":
        project = json.loads(path.read_bytes().rstrip(b"\0").decode("utf-8-sig"))
    else:
        raise ValueError("工程需要 SVP 或 VSQX 文件")
    tempos = project["time"]["tempo"]
    if (not tempos or tempos[0]["position"] != 0
            or any(not math.isfinite(t["bpm"]) or t["bpm"] <= 0 for t in tempos)
            or any(a["position"] >= b["position"] for a, b in zip(tempos, tempos[1:]))):
        raise ValueError("工程速度表无效")
    def seconds(position):
        previous, total, bpm = 0, 0, tempos[0]["bpm"]
        for tempo in tempos[1:]:
            if tempo["position"] > position:
                break
            total += (tempo["position"] - previous) / QUARTER * 60 / bpm
            previous, bpm = tempo["position"], tempo["bpm"]
        return total + (position - previous) / QUARTER * 60 / bpm
    solo = any(t.get("mixer", {}).get("solo") for t in project["tracks"])
    active = [t for t in project["tracks"] if not t.get("mixer", {}).get("mute")
              and (not solo or t.get("mixer", {}).get("solo"))
              and (t.get("groups") or t.get("mainGroup", {}).get("notes"))]
    if len(active) != 1:
        raise ValueError("请提供只含一条人声轨的工程与对应干声，不支持多声部混音")
    track = active[0]
    if track.get("groups"):
        raise ValueError("SVP 含引用组，请先在编辑器中合并到主音符组")
    ref = track.get("mainRef", {})
    notes = [dict(n, onset=n["onset"] + ref.get("blickOffset", 0),
                  pitch=n["pitch"] + ref.get("pitchOffset", 0)) for n in track["mainGroup"]["notes"]]
    notes.sort(key=lambda n: n["onset"])
    intervals = []
    for note in notes:
        if note["onset"] < 0 or note["duration"] <= 0 or not 0 <= note["pitch"] <= 127:
            raise ValueError("工程存在无效音符")
        if not note.get("phonemes", "").strip() and note.get("lyrics", "").strip() in ("", "-", "+"):
            raise ValueError("存在无歌词或延音符号，请先补充明确歌词／音素")
        start, end = seconds(note["onset"]), seconds(note["onset"] + note["duration"])
        if intervals and start < intervals[-1][1] - 1e-8:
            raise ValueError("工程中音符重叠，不适合单声部训练")
        intervals.append((start, end, note["pitch"]))
    if not intervals or intervals[-1][1] > 599.9:
        raise ValueError("工程为空或超过当前 10 分钟限制，请拆分工程")
    return dict(time=dict(tempo=tempos), tracks=[dict(mainGroup=dict(notes=notes))]), intervals
