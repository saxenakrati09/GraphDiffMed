# GraphDiffMed: Knowledge-Constrained Differential Attention with Pharmacological Graph Priors for Medication Recommendation

Recommending safe and effective medication combinations from electronic health records (EHRs) is a core clinical AI problem, yet it remains difficult because patient trajectories are long, noisy, and clinically heterogeneous. Existing methods typically excel at either temporal modeling across visits or pharmacological knowledge integration (e.g., drug-drug interactions, DDIs), but rarely achieve both while robustly suppressing noise. We present GraphDiffMed, a knowledge-constrained medication recommendation framework that unifies dual-scale Differential Attention v2 with DDI graph priors. Differential attention is applied at both intra-visit and inter-visit levels to filter spurious signals within encounters and across longitudinal history. In parallel, DDI structure is injected as an attention bias, anchoring representation learning to validated pharmacological relations rather than purely data-driven co-occurrence patterns. A post-hoc implementation audit showed that the intended intra-visit scalar graph bias becomes a uniform logit shift and is therefore ineffective under softmax, so empirical graph-bias gains originate from the inter-visit mechanism. Experiments on MIMIC-III and ablation studies show that this combination consistently improves recommendation quality and ranking over strong baselines while achieving a more favorable safety performance balance. We further find that the strongest-performing configuration uses only demographic auxiliary features under our experimental setting. Overall, GraphDiffMed demonstrates that combining noise-aware attention with explicit pharmacological priors yields more reliable and clinically meaningful medication recommendation.

## Features

- Construction of heterogeneous and homogeneous medical graphs
- Causal effect estimation between diagnoses, procedures, and medications
- Integration with molecular graph representations (using RDKit)
- End-to-end training and evaluation pipelines
- Support for large-scale EHR datasets (MIMIC-III, MIMIC-IV)
- Modular design for easy extension

## Installation

1. Clone the repository:
	```
	git clone <repo-url>
	cd GraphDiffMed
	```

2. Install dependencies:
	```
	pip install -r requirements.txt
	```

## Usage

### Data Preparation

Prepare your data in the `data/` directory. Use the same data as https://github.com/lixiang-222/CIDGMed. The framework expects preprocessed records and vocabulary files (see code for details).

### Relevance Matrix Construction

Run the relevance construction script:
```
python src/Relevance_construction.py --dataset mimic3
```

### Training and Evaluation

For running experiments with only diagnoses, procedures and medications:
```
bash src/run.sh
```
For running experiments with only diagnoses, procedures and medications + demographics:
```
bash src/run_demo.sh
```
For running experiments with only diagnoses, procedures and medications + labevents:
```
bash src/run_labevents.sh
```
For running experiments with only diagnoses, procedures and medications + labevents + demographics:
```
bash src/run_both.sh
```

## Project Structure

- `src/main.py` - Main entry point for training and evaluation
- `src/Relevance_construction.py` - Builds relevance matrices from EHR data
- `src/training.py` - Training and evaluation routines
- `src/util.py` - Utility functions and metrics
- `src/modules/` - Core model and graph construction modules

## Requirements

See `requirements.txt` for a full list. Key libraries:
- torch, torch-geometric
- numpy, pandas, scikit-learn
- rdkit, transformers
- cdt, dowhy
- matplotlib, tqdm, dill

## Citation

If you use this codebase, please cite the original paper (add citation here).

## License

This project is licensed under the MIT License.
