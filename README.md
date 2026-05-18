# Predictive Genomic Surveillance Framework

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyMC](https://img.shields.io/badge/PyMC-Bayesian_Inference-red)
![Data](https://img.shields.io/badge/Data->16M_Genomes-green)

An end-to-end bioinformatics pipeline that predicts the evolutionary success of novel viral variants by mathematically isolating epistatic drivers from genetic hitchhikers. 

**[Read the bioRxiv Pre-Print](#)** *(Insert your DOI link here later)*

## Project Overview
While traditional epidemiology relies on lagging indicators like hospitalizations or wastewater data, this project uses the viral genetic code itself as a leading indicator. By evaluating over 16 million global SARS-CoV-2 genomic sequences, this system utilizes a multivariate Bayesian Generalized Linear Model (GLM) to extract the true selection coefficients ($\boldsymbol{\theta}$) of specific mutations.

**Core Applications:**
* **Predictive Vaccine Design:** Allows for the forecasting of dominant variants to inform targeted booster shot design.
* **Data-Driven Containment:** Provides a mathematical risk metric for novel sequences to guide localized public health interventions.

## Technical Architecture

* **Big Data Extraction (MapReduce):** Implemented a streaming, $\mathcal{O}(1)$ memory architecture to parse 16+ million `.fasta`/`.tsv` records locally without RAM overflow. Extracts monthly global frequency distributions for targeted mutations.
* **Multivariate Bayesian Inference:** Designed a Bayesian GLM using `PyMC`. Applied L2 regularization via a Gaussian prior to effectively handle linkage disequilibrium, shrinking the coefficients of "passenger" mutations to zero while extracting the true weights of evolutionary "drivers."

## Quickstart

Clone the repository and set up the environment to run the analysis notebooks:

```bash
# 1. Clone the repo
git clone [https://github.com/yourusername/early-warning-genomics.git](https://github.com/yourusername/early-warning-genomics.git)
cd early-warning-genomics

# 2. Create the environment
conda create -n epi_model python=3.11
conda activate epi_model
pip install -r requirements.txt

# 3. Launch Jupyter to view the analysis
jupyter notebook