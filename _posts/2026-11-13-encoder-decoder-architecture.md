---
title: "The encoder-decoder architecture, and the shortcut that made it work"
description: "What an encoder-decoder network is, why its bottleneck loses detail, how skip connections and attention fix that, and where the architecture is still the standard."
date: 2026-11-13 17:00:00 +0100
category: Deep learning
published: false
---

An **encoder-decoder** is a two-stage network. It first compresses the input into an intermediate representation, then generates the output from that representation:

```text
input ──→ [Encoder] ──→ representation ──→ [Decoder] ──→ output
          compress,       (latent)          expand,
          understand                        generate
```

## Why two stages?

A network with only an encoder ends with something small: a class label, a score, an embedding. That is enough for classification.

But many tasks need a **structured output**: a sentence, an image, a segmentation mask, a piece of audio. The encoder can understand the input, but something has to unfold that understanding back into a full output. That is the decoder's job.

## The four parts

```text
        Encoder                Bottleneck               Decoder
input → compress step by step → most compact form → expand step by step → output
            │                                               ↑
            └────────────── (optional) shortcut ────────────┘
```

**1. Encoder.** Makes the input smaller and more abstract. In images, this is convolution plus downsampling.

**2. Bottleneck (latent representation).** The smallest, most abstract point of the network. All information has to pass through here.

**3. Decoder.** Turns the representation back into the form we need. In images, this is upsampling plus convolution; in text, it generates the output one word at a time.

**4. Shortcut connections (optional, but almost every modern version has them).** A plain encoder-decoder has a fundamental problem: **everything has to squeeze through the bottleneck, so fine details get lost.** The fix is a side path that lets the decoder look at the encoder's information directly:

- **Images:** **skip connections**, as in U-Net. Feature maps from each encoder level are passed straight to the matching decoder level.
- **Text:** **attention** (cross-attention). Every time the decoder generates a word, it can look back at every word of the input sentence.

In my view, this is **the most important improvement in the history of the encoder-decoder**. You can see how much difference it makes in `model.py` of my [Deep Learning Project 3 repository](REPO_LINK_HERE).

## Where encoder-decoders are used

They fit tasks where **both the input and the output are structured data**:

| Field | Task | Input → output | Typical models |
|---|---|---|---|
| Images | **Semantic segmentation** | image → mask | **U-Net**, SegNet, DeepLab |
| Images | Denoising, super-resolution, inpainting | image → image | U-Net variants |
| Images | Depth estimation | image → depth map | |
| Images | **Diffusion image generation** | noisy image → denoised image | the U-Net inside Stable Diffusion |
| Text | Machine translation | Chinese → English | original Transformer, T5 |
| Text | Summarisation | long text → short text | BART, T5 |
| Speech | Speech recognition | audio → text | Whisper |
| Multimodal | Image captioning | image → sentence | |
| Representation learning | Autoencoder | image → the same image | Autoencoder, VAE |

## Where it stands today

In **text generation**, encoder-decoders have largely been replaced by decoder-only models.

In **pixel-level image tasks** such as segmentation, they are still the standard. New models mostly swap out the parts inside, not the overall shape. The reason is simple: the output has to be as large as the input, one prediction per pixel, so you can hardly do without a decoder that expands the representation back to full resolution.
