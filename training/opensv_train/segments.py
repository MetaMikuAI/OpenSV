import math


def note_regions(intervals):
    regions = []
    for start, end, *_ in sorted(intervals):
        if regions and start <= regions[-1][1] + 1e-9:
            regions[-1] = (regions[-1][0], max(end, regions[-1][1]))
        else:
            regions.append((start, end))
    return regions


def annotated_segments(conditions, intervals, hop=.005):
    regions = note_regions(intervals)
    output = []
    for condition in conditions:
        rows = len(condition["counts"])
        if not rows or len(condition["phones"]) % rows:
            raise ValueError("Invalid phoneme feature dimensions")
        channels = len(condition["phones"]) // rows
        if not channels:
            raise ValueError("Empty phoneme features")
        frames = sum(condition["counts"])
        origin = condition["start"]
        for start, end in regions:
            first = max(0, math.ceil((start - origin) / hop - 1e-7))
            last = min(frames, math.floor((end - origin) / hop + 1e-7))
            if last <= first:
                continue
            phones, counts, languages = [], [], []
            cursor = 0
            for index, count in enumerate(condition["counts"]):
                keep = max(0, min(cursor + count, last) - max(cursor, first))
                if keep:
                    counts.append(keep)
                    phones.extend(condition["phones"][index * channels:(index + 1) * channels])
                    languages.append(condition["languages"][index])
                cursor += count
            output.append(dict(start=origin + first * hop, phones=phones, counts=counts,
                               languages=languages, noise=condition["noise"][first * 4:last * 4],
                               log_f0=condition["log_f0"][first:last]))
    return output
