# Generative AI Models for Images with Copyright-Free Training

Generative AI is transforming content creation, but its reliance on copyrighted training data raises important concerns about intellectual property. This repository contains the code, data, and analyses for our paper, which investigates whether state-of-the-art image generation performance can be achieved using only copyright-free data. We evaluate a publicly available model, Mitsua, against DALL·E 2, Stable Diffusion 2.1, and Juggernaut XL. By fine-tuning with small, curated image sets and using open-source techniques including LoRA, DreamBooth, and Textual Inversion, we show that the performance gap can be significantly reduced. We assess the models with FID, CLIP score, and a human preference study, and test the approach in three industry deployments. Our results highlight a practical path toward high-fidelity, domain-specific image generation that is also legally compliant.

## Repository Structure

```
1. MS COCO Benchmark/
├── 1. Dataset Creation/             Build the fine-tuning and benchmark image sets
├── 2. Generate Images with Models/  Generate benchmark images with the base models
├── 3. Fine-Tuning/                  Fine-tune with DreamBooth, LoRA, and Textual Inversion
├── 4. Scores/                       FID and CLIP scores, and statistical analyses of them
└── 5. Survey/                       Human preference study
2. Industry Deployments/
├── Convenience Chain/               Marketing image preference survey and ad A/B test
├── Fashion Retailer/                Product image preference survey
└── Beauty Startup/                  Ad A/B test
```

## 1. MS COCO Benchmark

The benchmark covers 76 MS COCO categories. Run the notebooks in this order:

| Step | Notebook | Description |
|---|---|---|
| 1 | `1. Dataset Creation/Dataset_Creation.ipynb` | Ranks images in each category by object size and by how well VGG16's predicted labels match the category. Keeps the top 20 images per category for fine-tuning and the top 100 as the benchmark. |
| 2 | `2. Generate Images with Models/Generate_Images_Using_Models.ipynb` | Generates one image per benchmark image from its COCO captions with Mitsua, Stable Diffusion 2.1, Juggernaut XL, and DALL·E 2. |
| 3 | `3. Fine-Tuning/Fine_Tuning.ipynb` | Captions the fine-tuning images and trains DreamBooth, LoRA, and Textual Inversion models for each category. Organized into one section per base model: Mitsua, Stable Diffusion 2.1, and Juggernaut XL. |
| 3 | `3. Fine-Tuning/Generate_Fine_Tuning.ipynb` | Generates the benchmark images with the fine-tuned models, in the same one-section-per-base-model layout. |
| 4 | `4. Scores/Scores.ipynb` | Computes FID and CLIP score per category for each model. |

Precomputed per-category scores are in `4. Scores/results/`:

| File | Contents |
|---|---|
| `baseline_FID.csv`, `baseline_CLIP.csv` | Mitsua, Stable Diffusion 2.1, Juggernaut XL, and DALL·E 2 |
| `finetuned_FID.csv`, `finetuned_CLIP.csv` | Mitsua fine-tuned with DreamBooth, LoRA, and Textual Inversion |
| `SOTA_FID.csv`, `SOTA_CLIP.csv` | Stable Diffusion 2.1 and Juggernaut XL fine-tuned with DreamBooth, LoRA, and Textual Inversion |
| `category_train_eval_clip_similarity.csv` | CLIP image similarity between each category's fine-tuning and benchmark images |

### Analyses

| Notebook | Analysis |
|---|---|
| `4. Scores/Heterogeneity.ipynb` | Share of categories in which each fine-tuning method improves on Mitsua's FID, and Spearman correlations between fine-tuning set representativeness (CLIP similarity to the benchmark images) and per-category FID improvement, with Holm–Bonferroni correction. |
| `4. Scores/Adaptation_Effect.ipynb` | Nine paired Wilcoxon signed-rank tests comparing each fine-tuned model's CLIP score with its own base model across the 76 categories, with a Bonferroni threshold of 0.05/9. |
| `5. Survey/analysis.ipynb` | Participants' rankings of the seven models: Friedman test, Kendall's W with a participant-level bootstrap 95% CI, Nemenyi post-hoc comparisons, and the Spearman correlation between the models' mean FID and their preference ranks. |
| `5. Survey/demographic.ipynb` | Participant gender, age, and country of residence. |

## 2. Industry Deployments

Partners are referred to by sector.

| Folder | Study | Notebooks |
|---|---|---|
| `Convenience Chain` | Pairwise preference survey (21 comparisons) between marketing images from Juggernaut XL and Mitsua fine-tuned with LoRA, plus an ad campaign A/B test report (`results/AB-Test-Raw Data-BP25.xlsx` and its PDF summary). | `survey.ipynb`: one-sample t-test and Wilcoxon signed-rank test on each participant's preference share. `demographics.ipynb`: participant demographics. `Mitsua_Trainer.ipynb`: the LoRA trainer used to adapt Mitsua for this deployment. |
| `Fashion Retailer` | Pairwise preference survey (13 comparisons) between product images from Juggernaut XL fine-tuned with LoRA and professional photographs. | `survey.ipynb`: same tests as above. `demographic.ipynb`: participant demographics. `JuggXL_Trainer.ipynb`: the LoRA trainer used to adapt Juggernaut XL for this deployment. |
| `Beauty Startup` | A/B test comparing an AI-generated and a human-shot ad creative on reach, impressions, clicks and conversions, summarized by cost per thousand impressions (CPM) and cost per click (CPC). | Data only, no notebook: the platform export (`PurelyBiome_ad_performance.csv`) and the summary workbook (`PurelyBiome_AB_Test.xlsx`). |

## Running the Code

**MS COCO pipeline and score analyses.** The notebooks in `1. MS COCO Benchmark/1.`–`4.` are written for Google Colab with a GPU and Google Drive mounted at `/content/gdrive`. Edit the path variables near the top of each notebook to point to your data. You will need:

- MS COCO 2014 instance and caption annotations for `train2014` and `val2014`, from [cocodataset.org](https://cocodataset.org).
- The Hugging Face models `Mitsua/mitsua-diffusion-one`, `stabilityai/stable-diffusion-2-1`, and `RunDiffusion/Juggernaut-XL-v9`.
- An OpenAI API key for DALL·E 2, set as `OPENAI_API_KEY` in `Generate_Images_Using_Models.ipynb`.
- For DreamBooth, replace `/content/diffusers/examples/dreambooth/train_dreambooth.py` with `3. Fine-Tuning/train_dreambooth.py`, as instructed in `Fine_Tuning.ipynb`.

**Survey and industry deployment analyses.** These notebooks run locally. Open each one from its own folder; it reads data from that folder's `results/` subfolder, and `5. Survey/analysis.ipynb` also reads FID scores from `4. Scores/results/`. They require pandas, NumPy, SciPy, matplotlib, and seaborn.

**Python packages** beyond the above: `pycocotools` throughout the MS COCO pipeline, `torch`, `torchvision`, `diffusers` and `transformers` for generation and fine-tuning, `tensorflow`/`keras` for the VGG16 ranking and the InceptionV3 FID, `scikit-image` for image loading and resizing, `spacy` with `en_core_web_md` for dataset creation, `openai` for DALL·E 2, and `scikit-learn` plus `statsmodels` for `Heterogeneity.ipynb`.

## Data and Privacy

Survey participants were recruited through Prolific. The released survey data is de-identified: Prolific participant IDs are replaced with study-specific codes (`P0001`, `P0002`, …), which link a participant's responses and demographics within one study but not across studies. IP addresses, locations, submission and response IDs, and completion codes have been removed.

Proprietary company training data and models were excluded. For the industry deployments, the partner-supplied images used for fine-tuning and the resulting fine-tuned models are not released; this repository contains only the survey responses, the aggregate ad performance reports, and the analysis code. The MS COCO benchmark is unaffected, since it is built from publicly available images.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
