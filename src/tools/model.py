from typing import Tuple

import tensorflow as tf
import toolz as tz
from src.tools import resnet_tools
from tensorflow import keras


def resnet_model(
    input_shape: Tuple[int] = (96, 96, 3), output_shape: Tuple[int] = (15, 1)
) -> keras.Model:
    X_input = keras.layers.Input(input_shape)

    fn_model = tz.compose_left(
        keras.layers.ZeroPadding2D((3, 3)),
        resnet_tools._one_convolution_block(64, (7, 7), (2, 2), "valid", "relu"),
        keras.layers.MaxPool2D((3, 3), (2, 2)),
        resnet_tools.projection_shortcut(num_filters=(64, 64, 64)),
        resnet_tools.identity_mapping(num_filters=(64, 64, 64)),
        resnet_tools.identity_mapping(num_filters=(64, 64, 64)),
        resnet_tools.identity_mapping(num_filters=(64, 64, 64)),
        resnet_tools.identity_mapping(num_filters=(64, 64, 64)),
        resnet_tools.identity_mapping(num_filters=(64, 64, 64)),
        resnet_tools.projection_shortcut(num_filters=(128, 128, 128)),
        resnet_tools.identity_mapping(num_filters=(128, 128, 128)),
        resnet_tools.identity_mapping(num_filters=(128, 128, 128)),
        resnet_tools.identity_mapping(num_filters=(128, 128, 128)),
        resnet_tools.identity_mapping(num_filters=(128, 128, 128)),
        resnet_tools.identity_mapping(num_filters=(128, 128, 128)),
        resnet_tools.identity_mapping(num_filters=(128, 128, 128)),
        resnet_tools.identity_mapping(num_filters=(128, 128, 128)),
        resnet_tools.projection_shortcut(num_filters=(512, 512, 512)),
        resnet_tools.identity_mapping(num_filters=(512, 512, 512)),
        resnet_tools.identity_mapping(num_filters=(512, 512, 512)),
        resnet_tools.identity_mapping(num_filters=(512, 512, 512)),
        resnet_tools.identity_mapping(num_filters=(512, 512, 512)),
        resnet_tools.identity_mapping(num_filters=(512, 512, 512)),
        keras.layers.AveragePooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(
            output_shape[0],
            activation="linear",
            kernel_initializer=keras.initializers.glorot_uniform(seed=42),
        ),
    )

    output = fn_model(X_input)

    return keras.Model(inputs=X_input, outputs=output, name="ResNet34")
