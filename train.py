import os
import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception, preprocess_input
from sklearn.model_selection import train_test_split

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class DataLoader:
    """
    Class to load and split data into training and validation sets.
    """
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def load_data(self):
        category_list = os.listdir(self.dataset_dir)
        filenames, categories = [], []
        
        for category in category_list:
            category_path = os.path.join(self.dataset_dir, category)
            for file_name in os.listdir(category_path):
                full_path = os.path.join(category_path, file_name)
                filenames.append(full_path)
                categories.append(category)

        df = pd.DataFrame({'filename': filenames, 'category': categories})
        df_train, df_valid = train_test_split(df, test_size=0.2, random_state=SEED)
        df_train.reset_index(drop=True, inplace=True)
        df_valid.reset_index(drop=True, inplace=True)
        
        return df_train, df_valid


class DatasetGenerator:
    """
    Class to create dataset generators for training and validation.
    """
    @staticmethod
    def create_train_dataset(df_train, input_size, batch_size=32):
        train_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        return train_gen.flow_from_dataframe(
            dataframe=df_train,
            x_col='filename',
            y_col='category',
            target_size=(input_size, input_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )

    @staticmethod
    def create_valid_dataset(df_valid, input_size, batch_size=32):
        valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        return valid_gen.flow_from_dataframe(
            dataframe=df_valid,
            x_col='filename',
            y_col='category',
            target_size=(input_size, input_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )


class ModelBuilder:
    """
    Class to create and compile the Xception-based model.
    """
    @staticmethod
    def build_model(input_size=150, learning_rate=0.001, size_inner=100, droprate=0.5):
        base_model = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=(input_size, input_size, 3)
        )
        base_model.trainable = False

        inputs = keras.Input(shape=(input_size, input_size, 3))
        base = base_model(inputs, training=False)
        vectors = keras.layers.GlobalAveragePooling2D()(base)
        inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
        drop = keras.layers.Dropout(droprate)(inner)
        outputs = keras.layers.Dense(10)(drop)

        model = keras.Model(inputs, outputs)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = keras.losses.CategoricalCrossentropy(from_logits=True)

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return model


class Trainer:
    """
    Class to handle model training and callbacks.
    """
    def __init__(self, model, train_ds, valid_ds, epochs, output_dir):
        self.model = model
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.epochs = epochs
        self.output_dir = output_dir
        self.callbacks = self._get_callbacks()
        self._create_class_mapping()

    def _create_class_mapping(self):
        class_mapping = {index: category for category, index in self.train_ds.class_indices.items()}
        with open(f"{self.output_dir}/class_mapping.json", "w", encoding="utf-8") as f:
            json.dump(class_mapping, f)
    
    def _get_callbacks(self):
        checkpoint = keras.callbacks.ModelCheckpoint(
            f'{self.output_dir}/xception_v1_best.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, verbose=1
        )
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        return [checkpoint, reduce_lr, early_stop]

    def train(self):
        history = self.model.fit(
            self.train_ds,
            epochs=self.epochs,
            validation_data=self.valid_ds,
            callbacks=self.callbacks
        )
        return history
    
if __name__ == '__main__':
    # Define parameters
    dataset_dir = "/dataset/Apparel images dataset new"
    input_size = 299
    batch_size = 32
    learning_rate = 0.001
    size_inner = 100
    droprate = 0.5
    epochs = 20
    output_dir = "/model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_loader = DataLoader(dataset_dir)
    df_train, df_valid = data_loader.load_data()

    # Generate datasets
    train_ds = DatasetGenerator.create_train_dataset(df_train, input_size, batch_size)
    valid_ds = DatasetGenerator.create_valid_dataset(df_valid, input_size, batch_size)

    # Build model
    model = ModelBuilder.build_model(input_size, learning_rate, size_inner, droprate)
    
    # Train model
    trainer = Trainer(model, train_ds, valid_ds, epochs, output_dir)
    history = trainer.train()