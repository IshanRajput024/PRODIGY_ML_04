# 🍔 Food Recognition and Calorie Estimation Model

This project aims to build a model that can classify food images and estimate their calorie content. It uses **InceptionV3**, a pre-trained Convolutional Neural Network (CNN), and fine-tunes it for food recognition. The model consists of two outputs:
1. **Food Classification**: Identifies the type of food.
2. **Calorie Estimation**: Estimates the calories for the given food item.

---

## 🎯 Features
- **Food Classification**: Recognizes food items from the Food-101 dataset.
- **Calorie Estimation**: Predicts the calories for each food item.
- **Transfer Learning**: Utilizes the InceptionV3 model pretrained on ImageNet to classify food items.

---

## 🛠️ Technologies Used
- **TensorFlow**: Deep learning framework used for building and training the model.
- **Keras**: High-level API used for building the neural network.
- **ImageDataGenerator**: For data augmentation and real-time image preprocessing.
- **InceptionV3**: Pretrained Convolutional Neural Network (CNN) model used for feature extraction.

---

## 📝 Dataset
The **Food-101 dataset** contains 101 different categories of food with images. You will need to manually download the dataset using the following link:
[Food-101 Dataset](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)

---

## 🚀 How to Run the Project

1. **Download the Dataset**:
   Download the Food-101 dataset from the provided link and extract it to a folder of your choice.

2. **Install Required Libraries**:
   Install the necessary Python libraries:
   ```bash
   pip install tensorflow numpy
data_path = "path/to/food-101/images"
python food_recognition_calorie_estimation.py

food_recognition_and_calorie_estimation/
📂 Repository Structure
├── food_recognition_calorie_estimation.py      # Main script to train the model
├── food_recognition_calorie_estimation_model.h5 # Saved model after training
├── README.md                                    # Project documentation
└── food-101/                                    # Folder where you store Food-101 dataset
🤔 Notes
Food Categories: In the model, we use 5 food categories for classification. Adjust the Dense(5, activation='softmax') layer as needed based on your classification setup.
Training Time: The training time depends on your hardware configuration, dataset size, and number of epochs.
