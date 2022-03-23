import itertools
from typing import Any, Dict, List, NewType, Tuple

import numpy as np
import pandas as pd
import toolz as tz

ParsedRawData = NewType(
    "ParsedRawData", Tuple[np.ndarray, Dict[str, float], Dict[str, float]]
)


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
) -> Dict[str, float]:
    pattern = f"_{coordinate}"
    return dict(
        [
            (z[0].replace(pattern, ""), z[1])
            for z in zip(names.ravel().tolist(), values.ravel().tolist())
            if z[0].endswith(pattern)
        ]
    )


def _parse_raw_data(image: np.ndarray, header: np.ndarray) -> ParsedRawData:
    coordinates, image = _extract_coordinates_and_image(image)
    fn_filter = _filter_coordinates(values=coordinates, names=header)

    x, y = fn_filter("x"), fn_filter("y")

    return image, x, y


def _convert_features_coordinates_to_dataframe(
    x_coodinates: dict, y_coordinates: dict
) -> pd.DataFrame:
    return pd.DataFrame(x_coodinates, index=["x"]).T.join(
        pd.DataFrame(y_coordinates, index=["y"]).T
    )
