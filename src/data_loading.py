import itertools
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import toolz as tz


def _parse_header(header_line: str, delimiter: str) -> np.ndarray:
    return np.reshape(
        np.array(list(map(lambda x: x.strip(), header_line.split(delimiter)))), (-1, 1)
    )


def read_raw_images(path: str, delimiter: str) -> np.ndarray:
    with open(path, "r") as f:
        header = _parse_header(f.readline(), delimiter)
        for line in itertools.islice(f, 1, None):
            yield np.reshape(np.array(line.split(delimiter)), (-1, 1)), header


def _extract_coordinates_and_image(image: np.ndarray) -> Tuple[np.ndarray]:
    coordinates = image[:30, :].astype(np.float32)
    image = np.reshape(
        np.array(image[-1, 0].split(" "), dtype=np.int32), newshape=(96, 96)
    )
    return coordinates, image


@tz.curry
def _filter_coordinates(
    coordinate: str, values: List[Any], names: List[str]
) -> Tuple[str, float]:
    pattern = f"_{coordinate}"
    return [
        (z[0].replace(pattern, ""), z[1])
        for z in zip(names.ravel().tolist(), values.ravel().tolist())
        if z[0].endswith(pattern)
    ]


def _parse_raw_data(image, header):
    coordinates, image = _extract_coordinates_and_image(image)
    fn_parser = _filter_coordinates(values=coordinates, names=header)

    x, y = fn_parser("x"), fn_parser("y")
    return image, pd.DataFrame(dict(x), index=["x"]).T.join(
        pd.DataFrame(dict(y), index=["y"]).T
    )
