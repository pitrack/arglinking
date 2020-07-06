# Heavily based off of https://github.com/luheng/lsgn

from typing import Any, Dict, List, Tuple
from overrides import overrides
import numpy as np

import torch # mostly for type annotations

from miso.metrics.metric import Metric
from miso.metrics.srl import compute_srl_f1

class ConllSrlScores(Metric):
  def __init__(self, vocab) -> None:
    self._scorer = Scorer()
    self._srl_vocab = vocab.get_index_to_token_vocabulary("labels")

  @overrides
  def __call__(self,
               predictions,
               gold,
               sentences,
               gold_conll_path: str = None):
    self._scorer.update(predictions["structures"],
                        gold["structures"],
                        sentences,
                        gold_conll_path)


  @overrides
  def get_metric(self, reset: bool = False) -> Tuple[float, float, float]: # ignore type?
    scores_dict = self._scorer.get_srl_metrics()
    if reset:
      self.reset()
    return scores_dict

  @overrides
  def reset(self):
    self._scorer.reset()


class Scorer:
  def __init__(self):
    self.all_gold_srl = []
    self.all_predictions = []
    self.sentences = []
    self.gold_conll_path = None

  def update(self,
             predictions,
             gold_srl,
             sentences,
             gold_conll_path):
    """
    predictions/gold_srl: List[Dict[Span, List[Tuple[Span, Label]]]]
    """

    self.gold_conll_path = gold_conll_path
    self.all_gold_srl.extend(gold_srl)
    self.all_predictions.extend(predictions)
    self.sentences.extend(sentences)

  def get_srl_metrics(self):
    (srl_p, srl_r, srl_f1,
     c_p, c_r, c_f1,
     ul_p, ul_r, ul_f1,
     label_confusions, comp_sents
    ) = compute_srl_f1(self.sentences, self.all_gold_srl,
                       self.all_predictions, self.gold_conll_path)
    metrics = {
               "conll_p": c_p,  # official stats
               "conll_r": c_r,
               "conll_f1": c_f1,
               "srl_precision": srl_p,  # unofficial stats
               "srl_recall": srl_r,
               "srl_f1": srl_f1,
               # "u_p": u_p, # unlabeled stats
               # "u_r": u_r,
               # "u_f": u_f,
    }
    return metrics

  def reset(self):
    self.all_gold_srl = []
    self.all_predictions = []
    self.sentences = []
