# HelmNet: Image Classification for Workplace Safety
An image classification model to to detect whether workers are wearing safety helmets in hazardous environments. The model aims to enhance workplace safety

## Overview
This project develops an image classification model to detect whether workers are wearing safety helmets in hazardous environments. Utilizing deep learning techniques, the model aims to enhance workplace safety by automating compliance monitoring, thereby reducing the risk of head injuries.

## Business Problem
Workplace safety in environments like construction sites and industrial plants is critical. Ensuring workers wear safety helmets is a primary safety measure. Manual monitoring for helmet compliance is inefficient and error-prone, especially in large operations. SafeGuard Corp seeks an automated image analysis system to detect helmet usage, improve safety enforcement, and minimize human error.

## Data
The dataset comprises **631 images**, equally split into two categories:
-   **With Helmet:** 311 images of workers wearing safety helmets.
-   **Without Helmet:** 320 images of workers not wearing safety helmets.

**Dataset Characteristics:**
-   **Variations in Conditions:** Images capture diverse environments (construction sites, factories) with varying lighting, angles, and worker postures.
-   **Worker Activities:** Workers are depicted in various actions (standing, using tools, moving) to ensure robust model learning.

## Approach
The project involved building and evaluating several image classification models:

1.  **Data Preprocessing:**
    *   Images were converted to grayscale (though RGB was used for the models).  (Note: The grayscale conversion was explored but not ultimately used for the models.)
    *   The dataset was split into training (80%), validation (10%), and testing (10%) sets.
    *   Image pixel values were normalized by dividing by 255 to scale them between 0 and 1.

2.  **Model Development:**
    *   **Model 1: Simple Convolutional Neural Network (CNN):** A custom CNN with convolutional, pooling, flatten, and dense layers.
    *   **Model 2: VGG-16 (Base):** Utilized the VGG16 pre-trained model (on ImageNet) with its convolutional layers frozen, followed by a Flatten layer and a single dense output layer.
    *   **Model 3: VGG-16 (Base + FFNN):** Similar to Model 2, but with an additional Feed-Forward Neural Network (Dense layers with Dropout) after the VGG16 base and before the output layer.
    *   **Model 4: VGG-16 (Base + FFNN + Data Augmentation):** This model incorporated data augmentation techniques (rotation, shift, shear, zoom) during training to improve generalization.

3.  **Model Evaluation:**
    *   **Metric:** Precision was chosen as the primary evaluation metric to minimize False Positives (predicting "With Helmet" when actually "Without Helmet"), which are critical for safety.
    *   Other metrics such as Accuracy, Recall, and F1 Score were also monitored.
    *   Confusion matrices were used for visual performance analysis.

## Results
All developed models achieved high performance on both training and validation sets. Specifically:

-   **Model 1 (Simple CNN):** Achieved high accuracy and precision, but showed a slight drop in performance from training to validation.
    *   Train: Accuracy ~0.994, Precision ~0.994
    *   Validation: Accuracy ~0.968, Precision ~0.970
-   **Model 2 (VGG-16 Base):** Achieved perfect scores on both training and validation sets.
    *   Train: Accuracy 1.0, Precision 1.0
    *   Validation: Accuracy 1.0, Precision 1.0
-   **Model 3 (VGG-16 Base + FFNN):** Also achieved perfect scores on both training and validation sets.
    *   Train: Accuracy 1.0, Precision 1.0
    *   Validation: Accuracy 1.0, Precision 1.0
-   **Model 4 (VGG-16 Base + FFNN + Data Augmentation):** Demonstrated perfect scores on both training and validation sets, and most importantly, on the unseen **test set**.
    *   Test: Accuracy 1.0, Precision 1.0

The final selected model (Model 4) achieved perfect accuracy and precision on the test set, indicating its strong ability to generalize to new, unseen images.

## Tools & Technologies
-   **Python**
-   **TensorFlow/Keras** (for building and training CNNs, VGG16)
-   **NumPy** (for numerical operations)
-   **Pandas** (for data manipulation)
-   **Matplotlib, Seaborn** (for data visualization)
-   **OpenCV (cv2)** (for image processing)
-   **Scikit-learn** (for data splitting and metrics)

## Key Learnings
-   All models performed exceptionally well, suggesting the dataset might be relatively straightforward or small.
-   The model incorporating **Data Augmentation** (Model 4) is considered the most robust and best choice for deployment, as it's less prone to overfitting and more likely to generalize to real-world variations.
-   The importance of choosing appropriate evaluation metrics (Precision in this case) for specific business contexts (minimizing False Positives in safety applications) was reinforced.
-   Transfer learning with pre-trained models like VGG16 can be highly effective even with relatively small datasets.
