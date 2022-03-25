from typing import Any, Callable, Dict, List, NewType, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf

# Data structure types
ArrayType = np.ndarray

TensorType = tf.Tensor

DataframeType = pd.DataFrame

DatasetType = tf.data.Dataset

# Generica function argument types
FnArgs = NewType("FnArgs", List[Any])

FnKwargs = NewType("FnKwargs", Dict[str, Any])

# Reader types
RawDataReader = Callable[[str, Optional[FnKwargs]], DataframeType]

# TransformerTypes
FnImageConverterType = Callable[
    [ArrayType, Optional[Union[FnArgs, FnKwargs]]], ArrayType
]

FnDataframeTransformerType = Callable[
    [DataframeType, Optional[Union[FnArgs, FnKwargs]]], DataframeType
]


FnTensorTransformerType = Callable[
    [TensorType, Optional[Union[FnArgs, FnKwargs]]], TensorType
]
