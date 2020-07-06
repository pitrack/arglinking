import json

from .conll_srl_scores import ConllSrlScores
from .gvdb_scores import GVDBScores

from miso.utils import logging


logger = logging.init_logger()


def dump_metrics(file_path: str, metrics, log: bool = False) -> None:
    metrics_json = json.dumps(metrics, indent=2)
    with open(file_path, "w") as metrics_file:
        metrics_file.write(metrics_json)
    if log:
        logger.info("Metrics: %s", metrics_json)
