import copy
from typing import Any, Callable, Tuple, Union

import tensorflow as tf
import toolz as tz
from tensorflow import keras


def _one_convolution_block(
    num_filters: int,
    filter_size: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    activation: str = None,
    training: bool = True,
    fn_initializer: Callable[[Any], Any] = keras.initializers.glorot_uniform,
) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    One convolution sub-component of the ResNet building blocks. Architecture consists
    of:

        Conv2D -> BatchNormalization -> Activation,

    or

        Conv2D -> BatchNormalization

    if no activation is specifiyed.

    :param num_filters: An interger representing the number of output filters in the
    convolution.

    :param filter_size: An interger or a tuple of 2 integers specifying the height and
    width of the middle convolutions filter used in the block.

    :param strides: An interger or a tuple of 2 integers specifying the strides of the
    convolutions along height and width.

    :param padding: A string specifying the type of padding. If 'valid' no padding is
    applyedl if 'same' equal padding is set left/right and top/down. If stride equals
    (1, 1) or 1, padding is set such as otput dimension matches input's.

    :param activation: A string defining the activation to be used when required. If
    None, no activation is used.

    :param training: A boolean used to lock parameters when not training the model in
    some of its layers.

    fn_initializer: Initializer for the kernel weights matrix (see keras.initializers).
    Defaults to 'random_uniform'.

    """
    batch_normalization = tz.partial(
        keras.layers.BatchNormalization(axis=3), training=training
    )
    convolution_2d = keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=filter_size,
        strides=strides,
        padding=padding,
        kernel_initializer=fn_initializer(seed=42),
    )

    if activation is None:
        fn = tz.compose(batch_normalization, convolution_2d)
    else:
        fn = tz.compose(
            keras.layers.Activation(activation), convolution_2d, batch_normalization
        )

    return fn


@tz.curry
def projection_shortcut(
    X: tf.Tensor,
    num_filters: Tuple[int],
    filter_size: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    activation: str = "relu",
    training: bool = True,
) -> tf.Tensor:
    """
    Residual network block which performs projection of the input a desired dimension in
    order to match output from the residual block when dimmension varies across blocks.

    :param X: The input tf.Tensor.

    :param num_filters: A tuple of 3 intergers representing the number of output filters
    in the convolution.

    :param filter_size: An interger or a tuple of 2 integers specifying the height and
    width of the middle convolution filter used in the block.

    :param strides: An interger or a tuple of 2 integers specifying the strides of the
    convolutions along height and width.

    :param activation: A string defining the activation to be used when required.

    :param training: A boolean used to lock parameters when not training the model in
    some of its layers.
    """
    shortcut_X = X
    filters_1, filters_2, filters_3 = num_filters

    fn_residual_mapping = tz.compose(
        _one_convolution_block(
            num_filters=filters_3,
            filter_size=(1, 1),
            strides=strides,
            padding="valid",
            activation=activation,
            training=training,
        ),
        _one_convolution_block(
            num_filters=filters_2,
            filter_size=filter_size,
            strides=(1, 1),
            padding="same",
            activation=activation,
            training=training,
        ),
        _one_convolution_block(
            num_filters=filters_1,
            filter_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            activation=activation,
            training=training,
        ),
    )

    fn_shortcut_projection = _one_convolution_block(
        num_filters=filters_3,
        filter_size=(1, 1),
        strides=strides,
        padding="valid",
        activation=activation,
        training=training,
    )

    return tz.pipe(
        [fn_shortcut_projection(shortcut_X), fn_residual_mapping(X)],
        keras.layers.Add(),
        keras.layers.Activation(activation),
    )


@tz.curry
def identity_mapping(
    X: tf.Tensor,
    num_filters: Tuple[int],
    filter_size: Union[int, Tuple[int]],
    activation: str = "relu",
    training: bool = True,
) -> tf.Tensor:
    """
    Residual network block.

    :param X: The input tf.Tensor.

    :param num_filters: A tuple of 3 intergers representing the number of output filters
    in the convolution.

    :param filter_size: An interger or a tuple of 2 integers specifying the height and
    width of the middle convolution filter used in the block.

    :param activation: A string defining the activation to be used when required.

    :param training: A boolean used to lock parameters when not training the model in
    some of its layers.
    """
    shortcut_X = X
    filters_1, filters_2, filters_3 = num_filters

    fn_residual_mapping = tz.compose(
        _one_convolution_block(
            num_filters=filters_3,
            filter_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            activation=activation,
            training=training,
            fn_initializer=keras.initializers.glorot_uniform,
        ),
        _one_convolution_block(
            num_filters=filters_2,
            filter_size=filter_size,
            strides=(1, 1),
            padding="same",
            activation=activation,
            training=training,
            fn_initializer=keras.initializers.glorot_uniform,
        ),
        _one_convolution_block(
            num_filters=filters_1,
            filter_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            activation=activation,
            training=training,
            fn_initializer=keras.initializers.glorot_uniform,
        ),
    )

    return tz.pipe(
        [shortcut_X, fn_residual_mapping(X)],
        keras.layers.Add(),
        keras.layers.Activation(activation),
    )
