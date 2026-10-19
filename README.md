# Self-Supervised Pretraining for Aerial Semantic Segmentation

PyTorch implementations of four self-supervised learning (SSL) methods, **SimCLR, BYOL, MAE and DINOv2**, compared on label-efficient semantic segmentation of drone imagery. Each pretrained encoder is paired with the same DeepLabV3 decoder and fine-tuned on only **60 labelled images**.

## Goal

Measure how much of a fully-supervised segmentation model's performance each SSL method can recover when only about one-fifth of the labelled data is available.

## Results

All encoders were pretrained for 50 epochs on 280 unlabelled images, then frozen. Only the decoder (and channel adapter, where present) was trained for 50 epochs on 60 labelled images. The fully-supervised baseline is a DeepLabV3-ResNet50 trained on 280 labelled images.

| Model | mIoU | Dice | Pixel Acc. | Labels used | % of fully-supervised mIoU |
|---|---|---|---|---|---|
| Fully supervised baseline | 0.553 | 0.676 | 0.852 | 280 | 100% |
| **SimCLR** | **0.291** | 0.413 | 0.622 | 60 | **52.6%** |
| BYOL | 0.250 | 0.364 | 0.579 | 60 | 45.1% |
| DINOv2 | 0.081 | 0.132 | 0.301 | 60 | 14.6% |
| MAE | 0.047 | 0.081 | 0.226 | 60 | 8.6% |

## Key Findings

- SimCLR and BYOL recover roughly half of the fully-supervised mIoU using about one-fifth of the labels.
- MAE and DINOv2 recover much less. The main cause is likely an unequal setup: SimCLR and BYOL continue from ImageNet-pretrained ResNet-50 weights, while MAE and DINOv2 train a compact ViT from scratch on only 280 images.
- Unfreezing the encoder made results worse with 60 labels, so all four methods use a frozen encoder.
- Class imbalance hurts rare and thin classes (door, fence-pole, dog, ar-marker). MAE and DINOv2 drop to zero IoU on six or more of the 24 classes, while SimCLR and BYOL degrade more gradually.
- Pretraining loss did not predict downstream quality. MAE had a healthy loss curve but the lowest mIoU.

## Data

Aerial (nadir) drone imagery with 24 semantic classes. The train/validation/test split is leakage-safe, verified disjoint by source-image key.

| Split | Images | Role |
|---|---|---|
| Train | 280 | Unlabelled SSL pretraining pool |
| Validation | 60 | Labelled fine-tuning set |
| Test | 60 | Monitoring and final evaluation |

## Implementation

| Method | Type | Backbone | Initialization |
|---|---|---|---|
| SimCLR | Contrastive (NT-Xent, temperature 0.5) | ResNet-50 | ImageNet-pretrained |
| BYOL | Negative-free, EMA target (decay 0.996), predictor head | ResNet-50 | ImageNet-pretrained |
| MAE | Masked reconstruction (75% masking, loss on masked patches) | Compact ViT (dim 384, 6 layers, patch 16) | From scratch |
| DINOv2 | Self-distillation (2 global + 6 local crops, EMA teacher, centering and sharpening) | Same compact ViT | From scratch |

**Augmentations (SimCLR/BYOL):** random-resized-crop, colour jitter, horizontal flip, vertical flip, and 90-degree rotation (the last two because nadir imagery has no fixed "up").

**Decoder:** the DeepLabV3 ASPP head is reused unchanged for all four methods.
- SimCLR/BYOL: 512x512 input, 2048-channel ResNet features go straight to the decoder.
- MAE/DINOv2: 224x224 input, 384-channel patch tokens are reshaped to a 2D grid and passed through a 1x1 conv adapter (384 to 2048 channels).

**Fine-tuning:** AdamW, decoder LR 1e-4, weight decay 1e-4, batch size 8, 50 epochs, frozen encoder.

**Evaluation:** mIoU, Dice, pixel accuracy, per-class IoU, normalized confusion matrices, and t-SNE of frozen encoder embeddings.

## Methodology Overview

1. Create a leakage-safe train/validation/test split.
2. Pretrain each encoder on the unlabelled pool for 50 epochs.
3. Attach the shared DeepLabV3 decoder and fine-tune on the 60 labelled images with the encoder frozen.
4. Compare all methods against the fully-supervised baseline using mIoU, Dice, pixel accuracy, per-class IoU, confusion matrices, and t-SNE of the frozen embeddings.

Developed on free-tier GPUs with Python, PyTorch, torchvision, NumPy, scikit-learn and Matplotlib.

## Limitations

- **Unequal comparison.** The ImageNet-initialized ResNets and the from-scratch ViTs do not start on equal footing, so the results do not show that one pretext task is better than another.
- **Different input resolutions.** 512x512 for SimCLR/BYOL versus 224x224 for MAE/DINOv2.
- **Test split used twice.** It selects the best fine-tuning checkpoint and reports final metrics, a simplification forced by the small labelled pool.
- **Undefined class.** One class (`conflicting`) is absent from the test split, so its IoU is undefined for every model.

## Future Work

1. Initialize MAE and DINOv2 from ImageNet-pretrained checkpoints to remove the initialization difference.
2. Use class-weighted losses or rare-class oversampling during fine-tuning.
3. Pretrain on a much larger unlabelled pool.