# train.py
import os
import numpy as np
from tensorflow.keras.applications import EfficientNetB4, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import wandb
from wandb.keras import WandbCallback
from sklearn.utils import class_weight
import argparse

# compile the model
def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB4(
        include_top=False,
        input_tensor=x,
        weights="imagenet",
        drop_connect_rate=config.drop_connect_rate,
    )

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    # top_dropout_rate = 0.2
    x = layers.Dropout(config.dropout, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNetB4")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate * 1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="provide location of images", type=str)

    args = parser.parse_args()

    # add the WandbCallback()
    early_stopping_monitor = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-2,
        patience=5,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    config_defaults = {
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "drop_connect_rate": 0.4,
        "dropout": 0.2,
        "seed": 777,
        "batch_size": 16,
        "valid_split": 0.2,
    }

    # Initialize a new wandb run
    wandb.init(project="killer-algae", config=config_defaults)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    # define model architecture
    IMG_SIZE = 400
    image_path = os.path.join(args.data)

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        image_path,
        validation_split=config.valid_split,
        subset="training",
        seed=config.seed,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=config.batch_size,
    )

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        image_path,
        validation_split=config.valid_split,
        subset="validation",
        seed=config.seed,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=config.batch_size,
    )

    img_augmentation = Sequential(
        [
            layers.RandomRotation(factor=0.15),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomFlip(),
            layers.RandomContrast(factor=0.1),
        ],
        name="img_augmentation",
    )

    NUM_CLASSES = 2

    def input_preprocess(image, label):
        label = tf.one_hot(label, NUM_CLASSES)
        return image, label

    train_data = train_data.map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    test_data = val_data.map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_labels = np.concatenate([np.argmax(y, 1) for x, y in train_data], axis=0)
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=train_labels
    )
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    model = build_model(num_classes=NUM_CLASSES)
    hist = model.fit(
        train_data,
        epochs=config.epochs,
        validation_data=test_data,
        verbose=2,
        class_weight=class_weights,
        callbacks=[early_stopping_monitor],
    )

    unfreeze_model(model)
    # epochs = 25
    hist = model.fit(
        train_data,
        epochs=config.epochs // 2,
        validation_data=test_data,
        verbose=2,
        class_weight=class_weights,
        callbacks=[WandbCallback(), early_stopping_monitor],
    )
    test_predictions = model.predict(test_data)
    top_pred_ids = test_predictions.argmax(axis=1)
    ground_truth_ids = np.concatenate([np.argmax(y, 1) for x, y in test_data], axis=0)
    wandb.log(
        {
            "confusion_matrix": wandb.plot.confusion_matrix(
                preds=top_pred_ids,
                y_true=ground_truth_ids,
                class_names=["NO HAB", "HAB"],
            )
        }
    )
    model.save("algae_cnn.h5")
