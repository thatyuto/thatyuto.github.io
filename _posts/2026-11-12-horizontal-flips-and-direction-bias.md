---
title: "Horizontal flips: a cheap fix for direction bias"
description: "Why a simple left-right flip is the only augmentation in my football shot detector, and why it has to be applied to every frame of a clip in the same way."
date: 2026-11-12 12:00:00 +0100
category: Computer vision
---

Data augmentation creates new training examples by changing existing ones slightly. In my football shot detector, I use just one augmentation, a **random horizontal flip**, and there are two reasons it is worth having.

## 1. More variety from the same videos

My dataset is small: a few hundred shot clips from 14 matches. Flipping a clip gives the model a slightly different version of the same moment, so it sees more variety without needing any new footage.

## 2. Direction should not matter

In a match, teams attack both goals, and they switch ends at half-time. Some shots go from left to right on screen, others from right to left.

With only a few hundred examples, it can happen by chance that more shots in the training set go towards one side. The model might then learn a shortcut such as "a shot means players moving to the right", which has nothing to do with shooting.

Flipping clips at random tells the model that **direction does not matter, but the shooting action does**.

## One flip per clip, not per frame

Each training example is 8 frames from a 4-second window. The flip has to be the same for all 8 frames:

```python
if self.split == "train" and random.random() < 0.5:
    frames = torch.flip(frames, dims=[3])   # flip width, same for all 8 frames
```

If each frame were flipped independently, the player and the ball would jump from one side of the pitch to the other within a single clip. That is not a real football situation, and it would destroy the motion the model is supposed to learn.

## Only during training

The flip is applied only to training clips, with a 50% chance. Validation and test clips are always left unchanged, so the evaluation reflects real, unmodified footage.
