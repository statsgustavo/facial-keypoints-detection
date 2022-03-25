import pytest
import tensorflow as tf
from src.models import resnet
from src.tools import helpers


def test_one_convolution_block(input_tensor):
    fn = resnet._one_convolution_block(10, (3, 3), (2, 2), "valid")
    assert callable(fn)

    output_tensor = fn(input_tensor)
    assert (1, 13, 13, 10) == tuple(output_tensor.shape)

    fn2 = resnet._one_convolution_block(10, (5, 5), (1, 1), "same")
    output_tensor_2 = fn2(output_tensor)
    assert (1, 13, 13, 10) == tuple(output_tensor_2.shape)


def test_projection_shortcut_full_call(input_tensor):
    output_tensor = resnet.projection_shortcut(
        X=input_tensor, num_filters=(5, 5, 5), filter_size=(3, 3), strides=(2, 2)
    )

    assert (1, 14, 14, 5) == tuple(output_tensor.shape)


def test_projection_shortcut_full_partial(input_tensor):
    fn_projection = resnet.projection_shortcut(
        num_filters=(5, 5, 5), filter_size=(3, 3), strides=(2, 2)
    )
    assert callable(fn_projection)

    output_tensor = fn_projection(input_tensor)
    assert (1, 14, 14, 5) == tuple(output_tensor.shape)


def test_identity_mapping_full_call(input_tensor):
    output_tensor = resnet.identity_mapping(
        X=input_tensor, num_filters=(3, 3, 3), filter_size=(3, 3)
    )

    assert tuple(input_tensor.shape) == tuple(output_tensor.shape)


def test_identity_mapping_partial_call(input_tensor):
    fn_identity_mapping = resnet.identity_mapping(
        num_filters=(3, 3, 3), filter_size=(3, 3)
    )
    assert callable(fn_identity_mapping)

    output_tensor = fn_identity_mapping(input_tensor)
    assert tuple(input_tensor.shape) == tuple(output_tensor.shape)


def test_resnet_model():
    fn = resnet.model()
    assert callable(fn)
    assert isinstance(fn, tf.keras.Model)
