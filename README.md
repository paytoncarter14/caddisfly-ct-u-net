# caddisfly-ct-u-net

This repository contains code for a U-Net based machine learning model to automatically segment silk glands from caddisfly micro CT imagery. The model utilizes Keras within R. The model can be run with the `caddisfly_ct_model.R` script.

# Directories

- `cropped`: PNG images of caddisfly micro CT scan slices.
- `traced`: The same slices in the `cropped` directory, but thresholded to pure black and white. The white pixels represent silk glands that were segmented by hand.
- `output-example`: Example output of the `caddisfly_ct_model.R` script. Each slice has a "cropped", "traced", and "predict" PNG.
