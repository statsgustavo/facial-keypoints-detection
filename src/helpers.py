from typing import Any, Dict, List

import hydra
import omegaconf


def load_metadata(folders: List[str], files: List[str]) -> Dict[str, Any]:
    overides_ = [f"+{a}={b}" for a, b in zip(folders, files)]

    with hydra.initialize_config_module(config_module="src.conf"):
        metadata = hydra.compose(
            overrides=overides_
        )

    return omegaconf.OmegaConf.to_object(metadata)
