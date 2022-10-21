# train.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import argparse
import wandb
from wandb.keras import WandbCallback

# tensorflow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB4, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping


def plot_hist(hist):
    fig = plt.figure(figsize=(20, 10))
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    return fig


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_array, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    # img = keras.preprocessing.image.load_img(img_path)
    # img = keras.preprocessing.image.img_to_array(img)
    img_array1 = img_array.squeeze()

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array1.shape[1], img_array1.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img_array1
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    t = f"True {label.numpy()} Predicted {np.argmax(model.predict(img_array))}"
    # display(Image(cam_path))
    # plt.imshow(image.numpy().astype(np.uint8).squeeze())
    cam = tf.keras.preprocessing.image.load_img(cam_path)
    cam = tf.keras.preprocessing.image.img_to_array(cam)
    return cam.astype(np.uint8).squeeze(), image.numpy().astype(np.uint8).squeeze(), t


# compile the model
def build_base_model(num_classes, lr):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    model = Sequential(
        [
            img_augmentation,
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(2, name="outputs"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
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
    parser.add_argument(
        "--img-size",
        help="image size used for training, only squares accepted",
        type=int,
    )

    args = parser.parse_args()

    IMG_SIZE = args.img_size
    batch_size = 8
    NUM_CLASSES = 2
    image_path = os.path.join(args.data)

    train_data = tf.keras.utils.image_dataset_from_directory(
        image_path,
        validation_split=0.4,
        subset="training",
        seed=777,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
    )

    val_data = tf.keras.utils.image_dataset_from_directory(
        image_path,
        validation_split=0.4,
        subset="validation",
        seed=777,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
    )

    img_augmentation = tf.keras.Sequential(
        [
            layers.RandomRotation(factor=0.2),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomFlip("horizontal"),
            layers.RandomContrast(factor=0.1),
            layers.RandomCrop(100, 100),
            layers.RandomZoom(0.2),
        ],
        name="img_augmentation",
    )

    def input_preprocess(image, label):
        label = tf.one_hot(label, NUM_CLASSES)
        # label
        return image, label

    train_data = train_data.cache().map(input_preprocess).prefetch(tf.data.AUTOTUNE)
    test_data = val_data.cache().map(input_preprocess).prefetch(tf.data.AUTOTUNE)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Initialise new run
    run = wandb.init(entity="algal-blooms-sweden", project="2022-HAB-Season")
    wandb_callback = wandb.keras.WandbCallback(log_weights=False)

    epochs = 100
    learning_rate = 1e-4
    config = wandb.config
    config.learning_rate = learning_rate
    config.max_epochs = epochs
    config.batch_size = batch_size
    config.image_size = IMG_SIZE

    my_data = wandb.Artifact("hab_dataset", type="raw_data")
    my_data.add_dir(image_path)
    run.log_artifact(my_data)

    # with strategy.scope():
    model = build_base_model(num_classes=NUM_CLASSES, lr=learning_rate)
    hist = model.fit(
        train_data,
        epochs=epochs,
        class_weight=None,
        validation_data=test_data,
        verbose=0,
        callbacks=[callback, wandb_callback],
    )

    a = np.concatenate([np.argmax(y, 1) for x, y in test_data], axis=0)
    pred = [np.argmax(i) for i in model.predict(test_data)]
    labels = ["No HAB", "HAB"]

    cm = wandb.plot.confusion_matrix(y_true=a, preds=pred, class_names=labels)
    wandb.log({"conf_mat": cm})

    fig = plot_hist(hist)
    wandb.log({"accuracy_plot": wandb.Image(fig)})

    # metrics table
    report = classification_report(a, pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    wandb.log({"metrics_table": wandb.Table(dataframe=df)})

    # gradcam images
    # last_conv_layer_name = "top_conv"
    last_conv_layer_name = list(
        filter(lambda x: isinstance(x, tf.keras.layers.Conv2D), model.layers)
    )[-1].name

    for image, label in test_data.unbatch().take(10):
        img_array = image.numpy()
        img_array = np.expand_dims(img_array, axis=0)
        label = tf.argmax(label.numpy())

        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        import matplotlib.cm as cm

        a, h, title = save_and_display_gradcam(img_array, heatmap)

        objs = [a, h]
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        fig.suptitle(title, fontsize=15)
        for i in range(len(ax)):
            ax[i].imshow(objs[i])
        wandb.log({"gradcam_example": wandb.Image(fig)})

    # Convert model to tflite
    save_path = "."
    tf.saved_model.save(model, save_path)

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(
        "."
    )  # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open("hab_model.tflite", "wb") as f:
        f.write(tflite_model)

    artifact = wandb.Artifact("mobile_model", type="model")
    artifact.add_file("hab_model.tflite")
    wandb.log_artifact(artifact)

    wandb.finish()
