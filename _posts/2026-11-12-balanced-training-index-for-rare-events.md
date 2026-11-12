---
title: "Training on rare events: positives, negatives and a 1:1 balance"
description: "How I turned match annotations into a list of training clips, and why the clips are balanced 1:1 even though real shots are extremely rare."
date: 2026-11-12 10:00:00 +0100
category: Machine learning
---

Shots are rare in football. In a 90-minute match, almost everything you see is passing, running and waiting. That makes a shot detector a classic **rare-event** problem, and it shapes how the training data has to be built.

## The training index

The first step is not to cut video, but to write down *which* clips to cut. I store this as a JSON Lines file, `windows.jsonl`, where each line describes one 4-second clip:

```json
{"game": "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley", "split": "train", "half": 1, "center_sec": 270.441, "label": 1, "event": "Shots on target"}
{"game": "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley", "split": "train", "half": 1, "center_sec": 347.458, "label": 1, "event": "Shots on target"}
```

`label: 1` is a shot clip and `label: 0` is a background clip. The file contains only text, no images. It works like a list of exam questions; the next step goes into the videos and fetches the frames for each one:

```text
Labels-v2.json        match annotations
      ↓  build the index
windows.jsonl         which clips to use (text only)
      ↓  read frames from the videos
training data         actual images
      ↓
model training
```

## Defining negatives

A background clip is centred at least **30 seconds away from every annotated event** in the same half. If a "background" clip contained part of a shot, it would confuse the model, because the same kind of footage would be labelled both ways.

## Why a 1:1 ratio?

For every shot clip, I sample exactly one background clip. In my 14 matches, that gives 304 shots and 304 background clips (207 + 207 for training).

The reason is that real footage is extremely unbalanced. The matches average about 22 shots each, while a 90-minute match has around 5,400 one-second positions. Only about **0.4%** of those are centred on a shot.

If I trained on that natural ratio, the model could take the easy way out: predict "no shot" for everything and still be right more than 99% of the time. Balancing the classes 1:1 removes that shortcut and forces the model to learn what a shot actually looks like.

## The catch

Balanced training does not mean balanced reality. When the model later runs over a full match, it sees roughly 250 background windows for every shot window. Even a small false-positive rate on background turns into many false alarms.

That is why the decision threshold is tuned on separate validation matches, and why the project is judged on event-level precision and recall over whole matches, not on accuracy over balanced clips.
