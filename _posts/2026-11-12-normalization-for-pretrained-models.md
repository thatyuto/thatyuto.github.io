---
title: "Why pre-trained models need the same normalization"
description: "From video frames to model input: why I normalize football frames with ImageNet statistics before fine-tuning a pre-trained ResNet."
date: 2026-11-12 11:00:00 +0100
category: Deep learning
---

Once I have a list of clips with timestamps, the next step is to turn each clip into something a model can read. My dataset class does exactly that: it reads the frames, resizes them, and normalizes them. The normalization step looks like a small detail, but it matters a lot when you start from a pre-trained model.

## From a clip to a tensor

For each 4-second clip, the dataset:

1. Picks **8 evenly spaced timestamps** across the window.
2. Reads the frame at each timestamp from the right half's video.
3. Resizes each frame to 224 × 224 and converts it from OpenCV's BGR to RGB.
4. Scales pixel values to [0, 1] and **normalizes** them.

The result for one clip is a tensor of shape `[8, 3, 224, 224]`: 8 frames, 3 colour channels, 224 × 224 pixels.

## Why ImageNet statistics?

My model is a ResNet18 that was pre-trained on ImageNet. During that pre-training, every input image was normalized with ImageNet's per-channel mean and standard deviation:

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
```

The network's weights were learned for inputs on that exact scale. If I fed it raw pixels instead, the first layers would receive values in a range they have never seen, and the features they have already learned would not transfer properly.

Using the same mean and standard deviation puts my football frames on the same scale as the original training data. That way the model can reuse what it already knows about edges, textures and shapes from the very first step.

## What about the content difference?

Football broadcasts look nothing like most ImageNet photos of dogs, cars and everyday objects. Normalization does not fix that, and it is not meant to. The difference in *content* is handled by fine-tuning: training continues on football clips, and the weights adapt to the new domain.

## A side benefit

Normalization also centres the input values around zero with a similar spread on each channel. That tends to make gradient-based training more stable, which helps when fine-tuning on a small dataset.
