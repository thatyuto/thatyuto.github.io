---
title: "Split by game, not by window"
description: "Why the train/test split for my football shot detector is done per match, and what goes wrong if you split random windows instead."
category: Machine learning
# Change to true (or delete this line) after rewriting it in your own words
published: false
---

In my [Shot Replay](/#projects) project, the model looks at short clips of football video and decides whether a shot happens in the middle of the clip. Each training example is a 4-second window with 8 frames, centred on a labelled moment in a SoccerNet match.

Before training anything, I had to decide how to split the data into train, validation and test sets. I went with **10 / 2 / 2 whole matches**, not a random split of windows. Here is why.

## The tempting option

The quick way is to put every window from all 14 matches into one list, shuffle it, and cut it 80/10/10. It looks fair: every set gets a mix of shots and background.

## Why it leaks

Windows from the same match are not independent:

- **Neighbouring windows overlap.** At inference I slide a window every second, so two neighbouring 4-second windows share three seconds of footage.
- **Same match, same look.** The same pitch, kit colours, lighting, camera angles and broadcast graphics appear in every window of a game.
- **Same players and teams.** A model can learn "this is what a shot looks like *in this match*" instead of what a shot looks like in general.

If windows from one match land in both training and test, the test set is no longer unseen data. The model gets rewarded for recognising a match it has already studied, and the numbers look better than they will on a new game.

## What the split looks like

```json
{
  "train": ["10 matches"],
  "val":   ["2 matches"],
  "test":  ["2 matches"]
}
```

Every window from a match goes wherever that match goes. The validation matches are used to pick the decision threshold; the test matches are touched once, at the end.

## The cost

Splitting by match means fewer, chunkier units, so the validation and test sets are small and the results are noisier. With only two test matches I can't claim much about generalisation. But a noisy honest number is more useful than a precise optimistic one.

## What I'll check next

As an exercise, I plan to rerun training with a random window split and compare the validation metrics. If they jump up, that gap is roughly how much the leak was flattering the model.

---

*Draft: rewrite this in your own words before publishing.*
