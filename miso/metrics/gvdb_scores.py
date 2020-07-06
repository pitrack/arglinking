# Adapted from conll_srl_scores
from typing import Any, Dict, List, Tuple
from overrides import overrides
import numpy as np

import torch # mostly for type annotations

from miso.metrics.metric import Metric
from miso.metrics.srl import compute_srl_f1

class GVDBScores(Metric):
  def __init__(self, vocab) -> None:
    self._scorer = Scorer()
    self._string_scorer = Scorer()
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
    self._string_scorer.update(predictions["strings"], gold["strings"], sentences, "")


  @overrides
  def get_metric(self, reset: bool = False) -> Tuple[float, float, float]: # ignore type?
    metrics = {}
    precision, recall, f1, c_p, c_r, c_f1, u_p, u_r, u_f = self._scorer.get_srl_metrics()
    metrics.update({
      "conll_p": c_p,
      "conll_r": c_r,
      "conll_f1": c_f1,
      "srl_f1": f1,
      })
    aggregate_gold = 0
    aggregate_predicted = 0
    aggregate_matched = 0
    aggregate_partial_matched = 0
    for role in self._srl_vocab.values():
      role_metrics = self._string_scorer.get_string_metrics(role)
      metrics[role+"s-f1"] = role_metrics[2]
      metrics[role+"p-f1"] = role_metrics[5]
      aggregate_gold += role_metrics[6]
      aggregate_predicted += role_metrics[7]
      aggregate_matched += role_metrics[8]
      aggregate_partial_matched += role_metrics[9]
      metrics[role+"gold"] = role_metrics[6]
      metrics[role+"pred"] = role_metrics[7]
      metrics[role+"matched"] = role_metrics[8]
      metrics[role+"partialmatched"] = role_metrics[9]
    metrics["AGGs-f1"] = self._string_scorer._print_f1(aggregate_gold, aggregate_predicted, aggregate_matched)[2]
    metrics["AGGp-f1"] = self._string_scorer._print_f1(aggregate_gold, aggregate_predicted, aggregate_partial_matched)[2]
    if reset:
      self.reset()
    return metrics

  @overrides
  def reset(self):
    self._scorer.reset()
    self._string_scorer.reset()


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
    self.sentences.append(sentences)

  def get_srl_metrics(self):
    (p, r, f1,
     c_p, c_r, c_f1,
     ul_p, ul_r, ul_f1,
     label_confusions, comp_sents
    ) = compute_srl_f1(self.sentences, self.all_gold_srl,
                       self.all_predictions, self.gold_conll_path)
    return (p, r, f1, c_p, c_r, c_f1, ul_p, ul_r, ul_f1)

  def get_string_metrics(self, role):
    """
    compute exact and approximate matches
    """
    sentences = self.sentences
    gold_srl = self.all_gold_srl
    predictions = self.all_predictions

    total_gold = 0  # total number of relations (pred->arg arcs) in gold for these sentences
    total_predicted = 0  # total number of predicted relations (pred->arg arcs) for these sentences
    total_matched = 0  # total number of correctly predicted (predicate, argument, role) triples for these sentences
    total_partial_matched = 0  # total number of correctly predicted (predicate, argument) pairs (role might be wrong) for these sentences

    # Compute unofficial F1 of SRL relations.
    for gold, prediction in zip(gold_srl, predictions):  # gold and predicted SRL for a given document
      gold_rels = 0  # number of relations (pred->arg arcs) in gold data for this sentence
      pred_rels = 0  # number of predicted relations (pred->arg arcs) for this sentence
      matched = 0  # number of correctly predicted (predicate, argument, role) triples for this sentence

      for pred_id, gold_args in gold.items():
        filtered_gold_args = [a for a in gold_args if a[0] == role]  # filter by role
        total_gold += len(filtered_gold_args)
        gold_rels += len(filtered_gold_args)
        if pred_id not in prediction:
          continue
        for a0 in filtered_gold_args:  # gold arguments for this predicate
          for a1 in prediction[pred_id]:  # predicted arguments for this predicate
            if a0[0] == a1[0]: # role matches
              if a0[1].strip() == a1[1].strip(): # string exact match
                total_matched += 1
                matched += 1
              if a0[1].strip() in a1[1].strip() or a1[1].strip() in a0[1].strip(): # partial match
                total_partial_matched += 1
      for pred_id, args in prediction.items():
        filtered_args = [a for a in args if a[0] == role]  # filter by role
        total_predicted += len(filtered_args)
        pred_rels += len(filtered_args)

    precision, recall, f1 = self._print_f1(total_gold, total_predicted, total_matched)
    p_prec, p_recall, p_f1 = self._print_f1(total_gold, total_predicted, total_partial_matched)
    return (precision, recall, f1, p_prec, p_recall, p_f1, total_gold, total_predicted, total_matched, total_partial_matched)

  def _print_f1(self, total_gold, total_predicted, total_matched, message=""):
    precision = 100.0 * total_matched / total_predicted if total_predicted > 0 else 0
    recall = 100.0 * total_matched / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

  def reset(self):
    self.all_gold_srl = []
    self.all_predictions = []
    self.sentences = []
