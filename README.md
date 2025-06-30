# Generative AI Models for Images with Copyright-Free Training

Generative AI is transforming content creation, but its reliance on copyrighted training data raises important concerns about intellectual property. This repository contains the code and experiments for our paper, which investigates whether state-of-the-art image generation performance can be achieved using only copyright-free data. We evaluate a publicly available model, Mitsua, against DALL·E 2, Stable Diffusion 2.1, and Juggernaut XL. By fine-tuning with small, curated image sets and using open-source techniques including LoRA, DreamBooth, and Textual Inversion, we show that the performance gap can be significantly reduced. Our results highlight a practical path toward high-fidelity, domain-specific image generation that is also legally compliant.

## Repository Overview

This repository is organized into four main folders, each corresponding to a key stage in our experimental pipeline:

1. **MS COCO Dataset Creation**  
   Scripts for preparing the MS COCO dataset, including filtering, formatting, and organizing images for downstream tasks.

2. **Generate Images with Models**  
   Notebooks for generating images using baseline and state-of-the-art text-to-image diffusion models, including Mitsua, DALL·E 2, Stable Diffusion 2.1, and Juggernaut XL.

3. **Fine-Tuning**  
   Fine-tuning workflows using LoRA, DreamBooth, and Textual Inversion to adapt models to specific image domains.

4. **Scores**  
   Code for evaluating generated images using FID and CLIP metrics.


