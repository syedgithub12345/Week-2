# Week-1 and Week-2 Work
---
4-week internship presented by Edunet Foundation, in collaboration with AICTE & Shell, focusing on Green Skills using AI technologies.

Waste Classification using Convolutional Neural Network (CNN)

This project repository contains all the work done in internship under the Edunet Foundation Internship in collaboration with AICTE & Shell.

> **Important Update:**  
> **Apologies for the confusion!**  
> I initially shared the wrong GitHub link. Please visit the updated link below to access the repository containing all of the work completed across all weeks of the project:  
> [Correct GitHub Repository Link](https://github.com/syedgithub12345/Week-1)
> 
> https://github.com/syedgithub12345/Week-1
> 
> I sincerely apologize for any inconvenience this may have caused, and I appreciate your understanding. Thank you for your time and support.

---

# Waste Classification using Convolutional Neural Network (CNN)

This project is part of the **Week 1**, **Week 2** milestones for the **Edunet Foundation Internship** in collaboration with AICTE & Shell. The task is to classify waste images into two categories: **Organic** and **Recyclable**. In this milestone, we completed the data preprocessing and initial setup for the project, including data visualization and the development of the base CNN model.

---
Work Done By- Syed Safi Hasnain Naqvi

## Week 1 Summary

### Tasks Completed:
- **Data Loading & Preprocessing**: Images were loaded and preprocessed from the provided dataset.
- **Data Visualization**: Visualized the distribution of waste categories (Organic vs Recyclable) in the dataset.
- **Model Setup**: Set up the initial stages for building a Convolutional Neural Network (CNN) for image classification.

---
## Week 2 Summary

In Week 2, we implemented data augmentation, model training, and performance evaluation for our CNN model. The key tasks completed include:

### Tasks Completed:

**Data Augmentation**: Implemented transformations like rescaling, rotation, zoom, and flipping to enhance model generalization.

**CNN Model Implementation**: Built a multi-layer Convolutional Neural Network for classification.

**Model Compilation & Training**: Used Adam optimizer and binary cross-entropy loss function.

**Evaluation**: Analyzed model performance with accuracy and loss metrics.
---

## Dataset

The dataset used for this project can be obtained from the following sources:
1. **Directly in the repository** (if included).
2. **Kaggle Dataset**: [Waste Classification Data on Kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data/data)

The dataset consists of images of waste items categorized as **Organic** and **Recyclable**.

---

## Technologies Used

- **Python 3.10**
- **TensorFlow** (for CNN model building)
- **Keras** (for deep learning layers)
- **OpenCV** (for image processing)
- **Matplotlib** (for data visualization)
- **Pandas** (for data handling)
- **TQDM** (for progress bar)

---

## Installation

Since there is **no requirements.txt** file provided, manually install the necessary libraries using:

```bash
pip install opencv-python pandas matplotlib tensorflow tqdm
```

Ensure that you're using **Python 3.10** or a compatible version of Python.

---

## Model Architecture
The implemented **CNN model** consists of:
1. **Three Convolutional Layers** with increasing filters (32, 64, 128) and ReLU activation.
2. **Max Pooling Layers** for feature reduction.
3. **Fully Connected Layers (Dense)** for classification.
4. **Dropout Regularization** to prevent overfitting.
5. **Sigmoid Activation** for binary classification.

### **Model Summary:**
```
Layer (type)                Output Shape             Param #   
-------------------------------------------------------------
conv2d_1 (Conv2D)           (None, 222, 222, 32)     896       
activation_1 (Activation)   (None, 222, 222, 32)     0         
max_pooling2d_1 (MaxPooling2D) (None, 111, 111, 32)  0         
conv2d_2 (Conv2D)           (None, 109, 109, 64)     18,496    
activation_2 (Activation)   (None, 109, 109, 64)     0         
max_pooling2d_2 (MaxPooling2D) (None, 54, 54, 64)    0         
conv2d_3 (Conv2D)           (None, 52, 52, 128)      73,856    
activation_3 (Activation)   (None, 52, 52, 128)      0         
max_pooling2d_3 (MaxPooling2D) (None, 26, 26, 128)   0         
flatten_1 (Flatten)         (None, 86528)            0         
dense_1 (Dense)             (None, 256)              22,151,424
activation_4 (Activation)   (None, 256)              0         
dropout_1 (Dropout)         (None, 256)              0         
dense_2 (Dense)             (None, 64)               16,448    
activation_5 (Activation)   (None, 64)               0         
dropout_2 (Dropout)         (None, 64)               0         
dense_3 (Dense)             (None, 2)                130       
activation_6 (Activation)   (None, 2)                0         
-------------------------------------------------------------
Total params: **22,261,250** (84.92 MB)
Trainable params: **22,261,250** (84.92 MB)
Non-trainable params: **0**
```

---

## Data Augmentation & Preprocessing
To improve model generalization, we applied **Image Augmentation** using `ImageDataGenerator`:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)
```

---

## Training the Model
We trained the model using the **Adam optimizer** with a batch size of **256** over **10 epochs**:
```python
hist = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)
```
### **Training Output:**
```
Epoch 1/10
89/89 ━━━━━━━━━ 3329s 37s/step - accuracy: 0.7278 - loss: 0.5522 - val_accuracy: 0.8754 - val_loss: 0.3641
Epoch 2/10
89/89 ━━━━━━━━━ 2978s 35s/step - accuracy: 0.8223 - loss: 0.4405 - val_accuracy: 0.9023 - val_loss: 0.2984
...
Epoch 10/10
89/89 ━━━━━━━━━ 2922s 34s/step - accuracy: 0.9456 - loss: 0.1983 - val_accuracy: 0.9512 - val_loss: 0.1478
```

---

## Performance Evaluation
The model achieved a validation accuracy of **95.12%** by the 10th epoch.

We visualized training performance using Matplotlib:
```python
import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

---

## How to Run the Project
1. **Prepare the Dataset:**
   - Download from Kaggle if not already in the repository.
   - Place it in the `dataset/` directory.
2. **Run Jupyter Notebook:**
   - Open `wasteclassification.ipynb` in **Jupyter Notebook** or **Google Colab**.
   - Run all cells for data loading, visualization, augmentation, model training, and evaluation.

---

## Project Files

- **wasteclassification.ipynb**: Jupyter notebook containing the implementation for Week 1, including:
  - Data loading and preprocessing
  - Data visualization for waste categories
 
  - `wasteclassification.ipynb`: Contains the full implementation for Week 2.
  - `dataset/`: Directory containing images.


---

## Acknowledgments
Special thanks to **Edunet Foundation, AICTE, and Shell** for providing this learning opportunity and **Kaggle** for the dataset.

---

🚀 **Stay tuned for Week 3 updates!**

---
