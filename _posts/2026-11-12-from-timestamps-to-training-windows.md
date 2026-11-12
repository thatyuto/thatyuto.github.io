---
title: "From annotation timestamps to training windows"
description: "How a millisecond timestamp becomes a training example, how to pick a clean background clip, and why each half of a match needs its own timeline."
date: 2026-11-12 09:00:00 +0100
category: Computer vision
---

Before training a video model, it is worth checking by hand that the data really means what you think it means. In my football shot detector, I start by verifying two examples: one **shot window** and one **background window**. Doing this forced me to understand two details that are easy to get wrong: how timestamps are stored, and why the two halves of a match are handled separately.

## Two kinds of windows

The model learns from short clips, each 4 seconds long. There are two types:

- **Shot windows (positives):** a clip centred on the moment a shot happens.
- **Background windows (negatives):** a clip where no shot happens, so the model learns what "nothing special" looks like.

## From milliseconds to seconds

To build a shot window, I look up the annotation and read its timestamp. In SoccerNet, the `position` field is stored as a string of **milliseconds**, so the first step is a simple conversion:

```python
sec = int(ann["position"]) / 1000   # "724000" -> 724.0 seconds
```

That number becomes the centre of the 4-second window.

## Picking a clean background clip

A background window should contain no shot at all. If a "negative" clip accidentally includes part of a shot, the model receives a contradictory signal and learns less.

So the rule is strict: a background centre must be **at least 30 seconds away from every annotated event in that half**. That includes not only shots but also goals, corners, fouls and every other labelled event, since all of them are moments of action rather than quiet play.

In practice, I walk through the half on a 1-second grid, keep every point that passes the 30-second rule, and then randomly sample as many of them as there are shots in that half:

```python
candidates = []
sec = WINDOW_SEC / 2                     # start where a full window fits
while sec <= upper:
    if all(abs(sec - e) >= NEG_MIN_GAP_SEC for e in events[half]):
        candidates.append(sec)
    sec += 1.0

picked = rng.sample(candidates, min(len(shots[half]), len(candidates)))
```

## Why each half is treated separately

In SoccerNet, each half is a **separate video file**, and its clock starts again from 0:00. A `position` of 300,000 ms means "5 minutes into *this* half", not 5 minutes into the match.

If I mixed the two halves on one timeline, a shot at 5:00 in the second half would be looked up in the first half's video, and the window would show completely different footage. Keeping the half number with every timestamp avoids that, and it also means the 30-second rule only compares events within the same half.

## Takeaway

Two small checks (milliseconds vs. seconds, and which half a timestamp belongs to) decide whether every training example shows the right footage. It is much cheaper to verify them on one positive and one negative window than to discover the problem after training.
