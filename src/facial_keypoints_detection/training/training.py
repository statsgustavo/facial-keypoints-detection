import fire
import tensorflow as tf
from src.facial_keypoints_detection.models import resnet
from src.facial_keypoints_detection.tools import helpers

from . import dataprep


def training():
    model_params = helpers.load_metadata(["models"], ["resnet"])["models"]

    training, validation, test = dataprep.load_and_parse_datasets(
        model_params["batch_size"]
    )

    model = resnet.model(input_shape=(96, 96, 1), output_shape=(30, 1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=model_params["learning_rate"]),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    model.fit(
        training,
        epochs=model_params["epochs"],
        validation_data=validation,
    )


def cli():
    fire.Fire({"training": training})


if __name__ == "__main__":
    cli()
