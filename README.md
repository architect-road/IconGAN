# IconGAN - Official Pytorch Implementation

**Design What You Desire: Icon Generation from Orthogonal Application and Theme Labels**

Yinpeng Chen, Zhiyu Pan, Min Shi, Hao Lu, Zhiguo Cao, Weicai Zhong

Abstract:*Generative adversarial networks,(GANs) have been trained to be professional artists able to create stunning artworks such as face generation and image style transfer. In this paper, we focus on a realistic business scenario: automated generation of customizable icons given desired mobile applications and theme styles. We first introduce a theme-application icon dataset, termed AppIcon, where each icon has two orthogonal theme and app labels. By investigating a strong baseline StyleGAN2, we observe mode collapse caused by the entanglement of the orthogonal labels. To solve this challenge, we propose IconGAN composed of a conditional generator and dual discriminators with orthogonal augmentations, and a contrastive feature disentanglement strategy is further designed to regularize the feature space of the two discriminators. Compared with other approaches, IconGAN indicates a superior advantage on the AppIcon benchmark. Further analysis also justifies the effectiveness of disentangling app and theme representations.*

### Environment
- Linux OS
- 1–8 high-end NVIDIA GPUs with at least 16 GB of memory.
- Python 3.9.7, torch 1.8.0
- CUDA Toolkit 11.1
- Python libraries: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3.`

### Dataset

### Train

### Test