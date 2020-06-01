# High-z Specs

This folder includes Spectre's intrinsic blue-side predictions for ULAS J1120+0641 and ULAS J1342+0928. The following code loades the predictions from `J1120_blue.npz` and `J1342_blue.npz`.

```
import numpy as np
data = np.load('J1120_blue.npz')

wavelength_bins = data['wave']
blue_side_continua = data['blue']
```
