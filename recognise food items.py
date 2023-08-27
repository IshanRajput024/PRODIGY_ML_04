import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Download and load the Food-101 dataset
# You might need to adjust the paths and download the dataset manually
data_path = "path/to/food-101/images"
train_data = tf.keras.utils.get_file("food-101", "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz", untar=True, cache_dir=data_path)

# Set up data augmentation and preprocessing
image_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    data_path + '/food-101/train',
    target_size=10,
    batch_size=30,
    class_mode='categorical'
)

# Load pre-trained InceptionV3 model without top layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification and regression head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
food_predictions = Dense(5, activation='softmax')(x)  # Adjust num_food_categories
calorie_predictions = Dense(1, activation='linear')(x)

model = Model(inputs=base_model.input, outputs=[food_predictions, calorie_predictions])

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss=['categorical_crossentropy', 'mean_squared_error'], metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)

# Save the model for future use
model.save("food_recognition_calorie_estimation_model.h5")
