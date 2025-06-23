<div align="center"> 
    <h1> UncertainSAM: Fast and Efficient Uncertainty Quantification of the Segment Anything Model</h1>
</div>

<div align="center"> 
<a href="https://arxiv.org/abs/2505.05049">
  <img src="https://img.shields.io/badge/ArXiv-2505.05049-red?style=flat&label=ArXiv&link=https%3A%2F%2Farxiv.org%2Fabs%2F2505.05049" alt="Static Badge" />
</a>
<a href="https://openreview.net/forum?id=G3j3kq7rSC">
  <img src="https://img.shields.io/badge/ICML-OpenReview-blue?style=flat&label=ICML&link=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3DG3j3kq7rSC" alt="Static Badge" />
</a>
<a href="https://greenautoml4fas.github.io/UncertainSAM/">
  <img src="https://img.shields.io/badge/Project_Page-green?style=flat&label=Github.io&link=https%3A%2F%2Fgreenautoml4fas.github.io%2FUncertainSAM%2F" alt="Static Badge" />
</a>
<a href="https://github.com/GreenAutoML4FAS/UncertainSAM">
  <img src="https://img.shields.io/badge/GitHub-Code-yellow?style=flat&link=https%3A%2F%2Fgithub.com%2FGreenAutoML4FAS%2FUncertainSAM" alt="Static Badge" />
</a>
</div>

---

<p style="font-style: italic; background-color: #f0f0f0; padding: 10px; display: inline-block;">
The introduction of the Segment Anything Model (SAM) has paved the way for numerous semantic segmentation applications. For several tasks, quantifying the uncertainty of SAM is of particular interest. However, the ambiguous nature of the class-agnostic foundation model SAM challenges current uncertainty quantification (UQ) approaches. This paper presents a theoretically motivated uncertainty quantification model based on a Bayesian entropy formulation jointly respecting aleatoric, epistemic, and the newly introduced task uncertainty. We use this formulation to train USAM, a lightweight post-hoc UQ method. Our model traces the root of uncertainty back to under-parameterised models, insufficient prompts or image ambiguities. Our proposed deterministic USAM demonstrates superior predictive capabilities on the SA-V, MOSE, ADE20k, DAVIS, and COCO datasets, offering a computationally cheap and easy-to-use UQ alternative that can support user-prompting, enhance semi-supervised pipelines, or balance the tradeoff between accuracy and cost efficiency.
</p>

---
Method

<p align="center">
<img src="assets/framework.png">
</p>

Training Objectives
<p align="center">
<img src="assets/training_objectives.png">
</p>


Samples

<p align="center">
<img src="assets/teaser.png">
</p>

Ablation
<p align="center">
<img src="assets/ablation.png">
</p>

Correlation
<p align="center">
<img src="assets/correlation.png">
</p>

Experiments
<p align="center">
<img src="assets/evaluation.png">
</p>

Runtime
<p align="center">
<img src="assets/runtime.png">
</p>



---

## Citation


If you use this code in your research, please cite the following paper:

```
@inproceedings{
  kaiser2025uncertainsam,
  title={Uncertain{SAM}: Fast and Efficient Uncertainty Quantification of the Segment Anything Model},
  author={Timo Kaiser and Thomas Norrenbrock and Bodo Rosenhahn},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=G3j3kq7rSC}
}
```
