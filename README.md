# Peak Learning for Denoising Mass Spectrometry Imaging Data

This repository contains the code for the analysis presented in the MSc Thesis titled 'Peak Learning for Denoising Mass Spectrometry Imaging Data' by Chris Butcher and Serkan Shentyurk.

## Getting Started

This project aims to find the insulin-related peaks that are created during the MSI technique and remove them. The notebooks included in this repository generate findings and figures for various chapters of the thesis.

## Notebooks

- `find_map.ipynb`: Generates findings and figures for Chapter 3 of the thesis. It creates the map of the islets of Langerhans by applying edge detection algorithm and various clustering algorithms.
- `v1.ipynb`: Generates findings and figures for Chapters 2, 4, and 6 of the thesis. It applies Pearson correlation analysis and then fits Gaussian Mixture Models. Finally, the output of the various methods are compared and several tests, including UMAP analysis and spatial distribution analysis to validate the insulin-related peaks are done.

## Running the Analysis

To run the analysis, follow these steps:

```bash
git clone https://github.com/serkanshentyurk/MSc_thesis_Peak_Learning/
cd MSc_thesis_Peak_Learning
conda env create -f environment.yml
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

If you wish to access the data, please send an e-mail to Serkan Shentyurk serkanshentyurk@hotmail.com or Melanie Nijs melanie.nijs@kuleuven.be

## Licence 

This project is licensed under the MIT License.

