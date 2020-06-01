# Fully probabilistic quasar continua predictions near Lyman-α with conditional neural spline flows.
***Authors:*** *David M. Reiman, John Tamanas, J. Xavier Prochaska, and Dominika Ďurovčíková*
<!---[\[arxiv\]](arXiv)[\[bibtex\]](bibtex)-->
[![arXiv](http://img.shields.io/badge/arXiv-2005.xxxxx-cd5c5c.svg)](arXiv)
[![bibtex](http://img.shields.io/badge/bibtex-4682b4.svg)](bibtex)

<!--- smaller image here, like 2 examples -->
<img src="/samples/sample_readme.png">

In [Fully probabilistic quasar continua predictions near Lyman-α with conditional neural spline flows](http://arxiv.org), we introduce *Spectre*, a conditional rational quadratic neural spline flow which models the distribution over plausible blue-side QSO continua (1190 Å ≤ λ<sub>rest</sub> < 1290 Å) given the red-side spectrum (1290 Å ≤ λ<sub>rest</sub> < 2900 Å).

In this repository, you'll find:  
  * [Additional reconstruction examples generated from our test set](/samples)
  * Two `.npz` files each containing 5000 blue-side predictions for each of [ULAS J1120+0641 and ULAS J1342+0928](<data/>)
  * An [evaluation notebook](<notebooks/evaluation.ipynb>) which loads the optimized Spectre model and predicts blue-side continua given red-side spectra

# Dependencies

Required python packages are listed in [environment.yml](environment.yml). To create a Conda environment with these dependencies use the following command:

```
conda env create -f environment.yml
```
# Data

Spectre consumes pre-processed continua from the 14th data release (DR14) of the Sloan Digital Sky Survey (SDSS) quasar catalog captured by the extended Baryon Oscillation Spectroscopic Survey (eBOSS). This repository contains one such datum for use in the [evaluation notebook](<notebooks/evaluation.ipynb>). The pre-processing code is available here https://github.com/DominikaDu/QSmooth. 

<!---
**arXiv:** TBA.  
**Abstract:** Measurement of the red damping wing of neutral hydrogen in quasar spectra provides a probe of the epoch of reionization in the early universe. Such quantification requires precise and unbiased estimates of the intrinsic continua near Lyman-α (Lyα), a challenging task given the highly variable Lyα emission profiles of quasars. Here, we introduce a fully probabilistic approach to intrinsic continua prediction. We frame the problem as a conditional density estimation task and explicitly model the distribution over plausible blue-side continua (1190 Å ≤ λrest < 1290 Å) conditional on the red-side spectrum (1290 Å ≤ λrest < 2900 Å) using normalizing flows. Our approach achieves state-of-the-art precision and accuracy, allows for sampling one thousand plausible continua in less than a tenth of a second, and can natively provide confidence intervals on the blue-side continua via Monte Carlo sampling. We measure the damping wing effect in two high-redshift quasars and estimate the volume-averaged neutral fraction of hydrogen from each, finding x_HI = 0.239 ± 0.053 for ULAS J1120+0641 (z = 7.09) and x_HI = 0.189 ± 0.031 for ULAS J1342+0928 (z = 7.54).  

***
-->
