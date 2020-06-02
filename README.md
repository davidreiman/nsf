# Fully probabilistic quasar continua predictions near Lyman-α with conditional neural spline flows.
***Authors:*** *David M. Reiman, John Tamanas, J. Xavier Prochaska, and Dominika Ďurovčíková*
<!---[\[arxiv\]](arXiv)[\[bibtex\]](bibtex)-->
[![arXiv](http://img.shields.io/badge/arXiv-2006.00615-cd5c5c.svg)](https://arxiv.org/abs/2006.00615)
<!---[![bibtex](http://img.shields.io/badge/bibtex-4682b4.svg)](bibtex)-->

<!--- smaller image here, like 2 examples -->
<img src="/samples/sample_readme.png">

In [Fully probabilistic quasar continua predictions near Lyman-α with conditional neural spline flows](https://arxiv.org/abs/2006.00615), we introduce *Spectre*, a conditional rational quadratic neural spline flow which models the distribution over plausible blue-side QSO continua (1190 Å ≤ λ<sub>rest</sub> < 1290 Å) given the red-side spectrum (1290 Å ≤ λ<sub>rest</sub> < 2900 Å).

In this repository, you'll find:  
  * [Additional reconstruction examples generated from our test set](/samples)
  * Two `.npz` files each containing 5000 blue-side predictions for each of [ULAS J1120+0641 and ULAS J1342+0928](<data/>)
  * An [inference notebook](<notebooks/inference.ipynb>) which loads the optimized Spectre model and predicts blue-side continua given red-side spectra

# Dependencies

Required python packages are listed in [environment.yml](environment.yml). To create a Conda environment with these dependencies use the following command:

```
conda env create -f environment.yml
```
# Data

Spectre consumes pre-processed continua from the 14th data release (DR14) of the Sloan Digital Sky Survey (SDSS) quasar catalog captured by the extended Baryon Oscillation Spectroscopic Survey (eBOSS). This repository contains one such datum for use in the [inference notebook](<notebooks/inference.ipynb>). The pre-processing code is available here: https://github.com/DominikaDu/QSmooth.
