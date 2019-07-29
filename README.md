# nn_utils
Utility functions required for building and training Neural Networks

### Preprocessing
* `get_random_eraser`: Implementation of Cutout preprocessing technique
  * Source: https://github.com/yu4u/cutout-random-erasing
* `standard_scaling`: Subtract by mean, and divide by standard deviation for every color channel
* `random_pad_crop`: Adds padding to the image and then crops it randomly

### Visualize
* `show_image`: Display single image using matplotlib
* `plot_model_history`: Plots model accuracy and loss increasing number of epochs
* `current_accuracy`: Calculates accuracy of the model for existing weights

### LR Utils (Learning Rate)
* `LR_Finder`: Calculate optimal LR based on multiple iterations
* `OneCycleLR`: Implementation of One-Cycle Learning Rate
  * Source: https://github.com/titu1994/keras-one-cycle
