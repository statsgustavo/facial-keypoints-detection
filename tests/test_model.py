import pytest
import tensorflow as tf
from src import model


def test_resnet_model():
    fn = model.resnet_model()
    assert callable(fn)
    assert isinstance(fn, tf.keras.Model)
