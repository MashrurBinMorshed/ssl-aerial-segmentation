# Self-Supervised Pretraining for Aerial Semantic Segmentation

A comparative study of **SimCLR, BYOL, MAE and DINOv2** for label-efficient semantic segmentation of drone imagery, using only **60 labelled images** (21.4% of the labels used by the fully-supervised baseline).

> University project, Department of Computer Science and Engineering, East West University, Dhaka, Bangladesh.

---

## Research Question

How much of a fully-supervised segmentation model's performance can each self-supervised learning (SSL) method recover when only about one-fifth of the labelled data is available?

## Key Results

All SSL encoders were pretrained for 50 epochs on 280 unlabelled aerial images, then frozen. Only the DeepLabV3 decoder was fine-tuned (50 epochs) on 60 labelled images.

| Model | mIoU | Dice | Pixel Acc. | Labels used | % of fully-supervised mIoU |
|---|---|---|---|---|---|
| Fully supervised (Part A baseline) | 0.553 | 0.676 | 0.852 | 280 | 100% |
| **SimCLR** | **0.291** | 0.413 | 0.622 | 60 | **52.6%** |
| BYOL | 0.250 | 0.364 | 0.579 | 60 | 45.1% |
| DINOv2 | 0.081 | 0.132 | 0.301 | 60 | 14.6% |
| MAE | 0.047 | 0.081 | 0.226 | 60 | 8.6% |

**Takeaways**

- SimCLR and BYOL recover roughly half of the fully-supervised mIoU using about one-fifth of the labels.
- MAE and DINOv2 recover much less. The report argues this is driven mainly by an **experimental asymmetry**, not by the pretext tasks themselves: SimCLR and BYOL continue from ImageNet-pretrained ResNet-50 weights, while MAE and DINOv2 train a compact ViT **from scratch** on only 280 images.
- Unfreezing the encoder **degraded** results under a 60-label budget, so all four methods use a frozen encoder.
- Severe class imbalance in the dataset made rare and thin classes (door, fence-pole, dog, ar-marker) very hard. MAE and DINOv2 collapsed to zero IoU on six or more classes.
- A low or well-converged pretraining loss did **not** predict good downstream segmentation.

## Dataset and Split

[Semantic Drone Dataset](http://dronedataset.icg.tugraz.at) (nadir aerial photography, 24 semantic classes). The train/validation/test split is leakage-safe and verified disjoint by source-image key.

| Split | Images | Role in this project |
|---|---|---|
| Train | 280 | Unlabelled SSL pretraining pool |
| Validation | 60 | Labelled fine-tuning set |
| Test | 60 | Monitoring and final evaluation |

## Methods

| Method | Type | Backbone | Initialization |
|---|---|---|---|
| SimCLR | Contrastive (NT-Xent) | ResNet-50 | ImageNet-pretrained |
| BYOL | Negative-free, EMA target | ResNet-50 | ImageNet-pretrained |
| MAE | Masked reconstruction (75% masking) | Compact ViT (dim 384, 6 layers, patch 16) | From scratch |
| DINOv2 | Self-distillation (multi-crop) | Same compact ViT | From scratch |

**Shared pipeline:** pretrained encoder, dense features, adapter (if needed), DeepLabV3 ASPP decoder, per-pixel class logits.

- SimCLR/BYOL run at 512x512 and feed 2048-channel features directly to the decoder.
- MAE/DINOv2 run at 224x224 and use a 1x1 convolutional adapter (384 to 2048 channels) before the decoder.

**Fine-tuning setup:** AdamW, decoder LR 1e-4, weight decay 1e-4, batch size 8, 50 epochs, frozen encoder.

## Repository Structure

| Notebook | Purpose | Kaggle |
|---|---|---|
| `01_data_prep.ipynb` | Leakage-safe split and data preparation | [link](https://www.kaggle.com/code/mashrur135/nb-01-data-prep) |
| `02_simclr_pretrain.ipynb` | SimCLR pretraining | [link](https://www.kaggle.com/code/mashrur135/nb-simclr) |
| `03_simclr_downstream.ipynb` | SimCLR fine-tuning and evaluation | [link](https://www.kaggle.com/code/mashrur135/nb-simclr-downstream) |
| `04_byol_pretrain.ipynb` | BYOL pretraining | [link](https://www.kaggle.com/code/mashrur135/nb-byol) |
| `05_byol_downstream.ipynb` | BYOL fine-tuning and evaluation | [link](https://www.kaggle.com/code/mashrur135/nb-byol-downstream) |
| `06_mae_pretrain.ipynb` | MAE pretraining | [link](https://www.kaggle.com/code/mashrur135/nb-mae) |
| `07_mae_downstream.ipynb` | MAE fine-tuning and evaluation | [link](https://www.kaggle.com/code/mashrur135/nb-mae-downstream) |
| `08_dinov2_pretrain.ipynb` | DINOv2 pretraining | [link](https://www.kaggle.com/code/mashrur135/nb-dinov2) |
| `09_dinov2_downstream.ipynb` | DINOv2 fine-tuning and evaluation | [link](https://www.kaggle.com/code/mashrur135/nb-dinov2-downstream) |
| `10_final_comparison.ipynb` | Cross-method comparison, per-class IoU, confusion matrices, t-SNE | [link](https://www.kaggle.com/code/mashrur135/final-comparison-438) |

The full write-up is in [`report.pdf`](report.pdf).

## How to Reproduce

1. Download the Semantic Drone Dataset and run `01_data_prep.ipynb` to create the split.
2. Run each method's pretraining notebook, then its downstream notebook (the notebooks were developed on Kaggle's free-tier GPUs).
3. Run `10_final_comparison.ipynb` to reproduce the tables and figures.

**Requirements:** Python, PyTorch, torchvision, NumPy, scikit-learn, Matplotlib.

## Limitations

These are stated openly because they affect how the results should be read.

- **Unequal comparison.** SimCLR/BYOL start from ImageNet weights; MAE/DINOv2 are trained from scratch on 280 images (roughly 400 optimizer steps for DINOv2). The results do not show that one pretext task is better than another.
- **Different resolutions.** MAE/DINOv2 operate at 224x224 versus 512x512 for SimCLR/BYOL.
- **Test split used twice.** The test split was used both to select the best fine-tuning checkpoint and to report final metrics, a simplification forced by the small labelled pool.
- **Class imbalance.** One class (`conflicting`) is absent from the test split, so its IoU is undefined for every model.
- **DINOv2 instability.** Pretraining loss rose again once teacher temperature warmup finished, so the training budget was likely insufficient. A bug where the teacher network was left in training mode (dropout active) was found and fixed during development.

## Future Work

1. Initialize MAE and DINOv2 from official ImageNet-pretrained checkpoints to remove the initialization confound.
2. Use class-weighted losses or rare-class oversampling during fine-tuning.
3. Pretrain on a much larger unlabelled pool.

## References

1. T. Chen et al., "A simple framework for contrastive learning of visual representations," ICML, 2020.
2. J.-B. Grill et al., "Bootstrap your own latent," NeurIPS, 2020.
3. K. He et al., "Masked autoencoders are scalable vision learners," CVPR, 2022.
4. M. Oquab et al., "DINOv2: Learning robust visual features without supervision," arXiv:2304.07193, 2023.
5. M. Caron et al., "Emerging properties in self-supervised vision transformers," ICCV, 2021.
6. L.-C. Chen et al., "Rethinking atrous convolution for semantic image segmentation," arXiv:1706.05587, 2017.
7. Graz University of Technology, "Semantic Drone Dataset."
