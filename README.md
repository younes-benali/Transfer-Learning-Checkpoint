# Cat and Dog Image Classification using VGG16

This notebook demonstrates a simple image classification task to distinguish between cats and dogs using a pre-trained VGG16 model as a feature extractor.

## Project Overview

The goal of this project is to build and train a convolutional neural network (CNN) to classify images as either 'cat' or 'dog'. We leverage the power of transfer learning by using a pre-trained VGG16 model and fine-tuning it for our specific task.

## Dataset

The dataset used is the 'cat-and-dog' dataset available on KaggleHub. It consists of two main splits:
- `training_set`: Contains images for training the model.
- `test_set`: Contains images for evaluating the model's performance.

## Key Components

1.  **Data Download**: The dataset is downloaded from KaggleHub.
2.  **Data Loading and Preprocessing**: Images are loaded efficiently using `ImageDataGenerator` from Keras, which handles batching, resizing, and pixel value rescaling (to 0-1 range) on the fly, significantly reducing RAM usage compared to loading all images into memory.
    -   Images are resized to `(224, 224)` pixels, which is the expected input size for the VGG16 model.
    -   Batch size is set to `16` for training and evaluation.
3.  **Model Architecture**: A transfer learning approach is employed:
    -   **VGG16 Base Model**: We use the `VGG16` model pre-trained on ImageNet, without its top classification layers (`include_top=False`). The weights of this base model are frozen (`base_model.trainable = False`) to use it purely as a feature extractor.
    -   **Custom Classification Head**: A custom classification head is added on top of the VGG16 base model, consisting of:
        -   A `Flatten` layer to convert the 2D feature maps into a 1D vector.
        -   A `Dense` layer with 128 units and `relu` activation.
        -   A final `Dense` layer with 1 unit and `sigmoid` activation for binary classification (cat or dog).
4.  **Model Compilation**: The model is compiled with:
    -   `optimizer='adam'`
    -   `loss='binary_crossentropy'` (suitable for binary classification)
    -   `metrics=['accuracy']`
5.  **Model Training**: The model is trained for `3` epochs using the `train_generator` and validated against the `test_generator`. `steps_per_epoch` and `validation_steps` are calculated to ensure the generators provide sufficient data for each epoch.
6.  **Model Evaluation**: The trained model's performance is evaluated on the test set using the `test_generator` to calculate the final test loss and accuracy.
