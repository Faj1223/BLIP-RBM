
Improving Image-Text Generation with a Restricted Boltzmann Machine (RBM) on BLIP
Author
Fortuné HOUESSOU
Under the supervision of Mr. Jérôme Lacaille
2025

Project Overview
The convergence of computer vision and natural language processing has led to the development of Vision-Language Models (VLMs), such as BLIP (Bootstrapped Language-Image Pretraining). However, these models often produce overly simplified textual descriptions of images. This project aims to explore the integration of a Restricted Boltzmann Machine (RBM) to enhance the quality of descriptions generated by BLIP.

Objectives
The main objective of this research is to improve the text representations generated by BLIP by leveraging an RBM to capture richer relationships between text tokens and visual representations. Specifically, the goals are as follows:

Introduce an RBM after BLIP's encoder to refine the embeddings.
Compare the quality of descriptions before and after this enhancement.
Evaluate the results using various metrics such as Cosine Distance, Semantic Similarity, Perplexity, and Shannon Entropy.
Propose a custom metric for evaluating improvements (Statistical Variance of Generated Texts).
Assess model performance on different datasets and tasks such as image classification and image-text matching.
Methodology
1. BLIP-RBM Model
BLIP combines a contrastive model with a generative model. We aim to insert an RBM between the encoder and decoder to modify the latent representation and capture more information before generating the text.

2. Restricted Boltzmann Machine (RBM)
An RBM is a probabilistic model consisting of:

A visible layer 
𝑣
v (feature vectors extracted from BLIP).
A hidden layer 
ℎ
h that learns underlying representations.
Weighted connections between these layers, represented by 
𝑊
W.
The training process is unsupervised and is performed using Contrastive Divergence, a gradient approximation method.

3. Evaluation Metrics
We will use several metrics to compare the quality of descriptions generated before and after adding the RBM:

Perplexity: A measure of the fluency of the generated text.
BLEU Score: Evaluates the similarity between the generated text and a human reference.
Cosine Similarity: Compares the vector representations of the generated sentences.
Work Plan
In-depth study of BLIP and RBMs (Literature Review).
Implementation of the RBM and integration with BLIP.
Experimentation and evaluation of the results.
Writing the final report.
Conclusion
This project aims to enhance the quality of descriptions generated by BLIP by incorporating an RBM into the pipeline. Using appropriate metrics, we hope to demonstrate a significant improvement over the original BLIP model.

Installation
To get started with the project, clone the repository and install the necessary dependencies:

bash
Copier
Modifier
git clone https://github.com/yourusername/BLIP-RBM.git
cd BLIP-RBM
pip install -r requirements.txt
Requirements
Python 3.8+
PyTorch
NumPy
Transformers
Usage
After setting up the environment, you can run the model using the following command:

bash
Copier
Modifier
python run_model.py
This will start the process of training and evaluation. For more detailed usage instructions, please refer to the documentation provided in the docs/ directory.

License
This project is licensed under the MIT License - see the LICENSE file for details.
