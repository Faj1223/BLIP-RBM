# Improving Image-Text Generation with a Restricted Boltzmann Machine (RBM) on BLIP

## Project Overview

The convergence of computer vision and natural language processing has led to the development of Vision-Language Models (VLMs), such as BLIP (Bootstrapped Language-Image Pretraining). However, these models often produce overly simplified textual descriptions of images. This project explores the integration of a Restricted Boltzmann Machine (RBM) to enrich BLIP’s latent representations and improve the quality of its generated descriptions.

## Objectives

The main objective of this research is to enhance the text representations produced by BLIP by leveraging an RBM to capture richer relationships between text tokens and visual features. More specifically, the goals are:

- Introduce an RBM after BLIP’s encoder to refine latent embeddings.
- Compare the quality of descriptions before and after the RBM enhancement.
- Evaluate the results using metrics such as Cosine Similarity, Semantic Similarity, Perplexity, and Shannon Entropy.
- Propose a custom evaluation metric (Statistical Variance of Generated Texts).
- Assess model performance on tasks such as image-text matching and image classification.

## Methodology

### 1. **BLIP-RBM Model**

BLIP combines a contrastive objective with a generative decoder. In this project, an RBM is inserted between the encoder and the decoder to modify and enrich the latent representation before caption generation.

### 2. **Restricted Boltzmann Machine (RBM)**

An RBM is a probabilistic model composed of:

- A visible layer \(v\) (BLIP’s encoded features),
- A hidden layer \(h\) that learns latent structure,
- A weight matrix \(W\) connecting both layers.

The RBM is trained in an unsupervised manner using Contrastive Divergence, a commonly used approximation of the gradient.

### 3. **Evaluation Metrics**

To compare the captions generated before and after inserting the RBM, several metrics are used:

- **Perplexity** — measures fluency and language quality.
- **BLEU Score** — evaluates similarity to human-reference captions.
- **Cosine Similarity** — compares semantic embeddings of generated captions.

A custom metric based on **textual variance** is also proposed to quantify descriptive richness.

## Work Plan

1. Literature review: BLIP, RBMs, and hybrid VLM architectures.
2. Implementation of the RBM and integration into the BLIP pipeline.
3. Experiments and evaluation on multiple datasets.
4. Final report and analysis.

## Conclusion

This project investigates whether enriching BLIP’s latent representation with an RBM can lead to more informative and diverse image descriptions. Using several evaluation metrics, the aim is to demonstrate quantifiable improvements over the original BLIP model.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/BLIP-RBM.git
cd BLIP-RBM
pip install -r requirements.txt
