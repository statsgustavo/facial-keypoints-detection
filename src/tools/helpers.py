from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import omegaconf


def load_metadata(folders: List[str], files: List[str]) -> Dict[str, Any]:
    overides_ = [f"+{a}={b}" for a, b in zip(folders, files)]

    with hydra.initialize_config_module(config_module="src.conf"):
        metadata = hydra.compose(overrides=overides_)

    return omegaconf.OmegaConf.to_object(metadata)


def _compute_one_dimension_size(
    input_dim_size: int, filter_dim_size: int, stride: int, padding: int
) -> int:
    return np.floor(
        ((input_dim_size + 2 * padding - filter_dim_size) / stride) + 1
    ).astype(np.int32)


def _same_convolution_padding(filter_shape):
    filter_height, filter_width = filter_shape
    same_padding_height, same_padding_width = (
        (filter_height - 1) / 2,
        (filter_width - 1) / 2,
    )
    return same_padding_height, same_padding_width


def convolution_output_dimension(
    input_shape: Tuple[int],
    filter_shape: Tuple[int],
    stride: Tuple[int],
    padding: Tuple[int],
    num_filters: int,
) -> Tuple[int]:
    input_height, input_width = input_shape
    filter_height, filter_width = filter_shape
    stride_height, stride_width = stride
    padding_height, padding_with = padding

    output_height = _compute_one_dimension_size(
        input_height, filter_height, stride_height, padding_height
    )

    output_width = _compute_one_dimension_size(
        input_width, filter_width, stride_width, padding_with
    )

    return (output_height, output_width, num_filters)
