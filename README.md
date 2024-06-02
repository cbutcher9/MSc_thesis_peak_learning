# Peak Learning for Denoising Mass Spectrometry Imaging Data

This repository contains the code for the analysis presented in the MSc Thesis titled 'Peak Learning for Denoising Mass Spectrometry Imaging Data' by Chris Butcher and Serkan Shentyurk.

## Getting Started

This project aims to find the insulin-related peaks that are created during the MSI technique and remove them. The notebooks included in this repository generate findings and figures for various chapters of the thesis.

## Notebooks

- `pearson_approach.ipynb`: Generates findings and figures for the Pearson correlation approach in Chapter 5 of the thesis.
- `spearman_approach.ipynb`: Generates findings and figures for the Spearman correlation approach in Chapter 5 of the thesis.

## Running the Analysis

To run the analysis, follow these steps:

```bash
git clone https://github.com/cbutcher9/MSc_thesis_peak_learning/
cd MSc_thesis_peak_learning
conda env create -f requirements.txt
conda activate thesis_final
```
Then, start Jupyter Notebook:
```
jupyter notebook
```
Alternatively, you can use VS Code:
```
code
```

## Usage

Once the Jupyter Notebook server is running, open the desired notebook (find_map.ipynb or v1.ipynb) to explore the analysis and results.

## Data Availability

If you wish to access the data, please send an e-mail to Serkan Shentyurk clkbutcher@btinernet.com or Melanie Nijs melanie.nijs@kuleuven.be

## Licence 

This project is licensed under the MIT License.

