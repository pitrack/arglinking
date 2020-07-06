# Evaluation util functions for PropBank SRL.

import codecs
from collections import Counter
import operator
import os
from os.path import join
import subprocess

_SRL_CONLL_EVAL_SCRIPT  = "miso/metrics/srl/run_conll_eval.sh"
ignore_errors = True

def split_example_for_eval(example):
  """Split document-based samples into sentence-based samples for evaluation.

  Args:
    example:
  Returns:
    Tuple of (sentence, list of SRL relations)
  """
  sentences = example["sentences"]
  num_words = sum(len(s) for s in sentences)
  word_offset = 0
  samples = []
  for i, sentence in enumerate(sentences):
    srl_rels = {}
    ner_spans = []  # Unused.
    for r in example["srl"][i]:
      pred_id = r[0] - word_offset
      if pred_id not in srl_rels:
        srl_rels[pred_id] = []
      srl_rels[pred_id].append((r[1] - word_offset, r[2] - word_offset, r[3]))
    samples.append((sentence, srl_rels, ner_spans))
    word_offset += len(sentence)
  return samples


def evaluate_retrieval(span_starts, span_ends, span_scores, pred_starts, pred_ends, gold_spans,
                       text_length, evaluators, debugging=False):
  """
  Evaluation for unlabeled retrieval.

  Args:
    gold_spans: Set of tuples of (start, end).
  """
  if len(span_starts) > 0:
    sorted_starts, sorted_ends, sorted_scores = zip(*sorted(
        zip(span_starts, span_ends, span_scores),
        key=operator.itemgetter(2), reverse=True))
  else:
    sorted_starts = []
    sorted_ends = []
  for k, evaluator in evaluators.items():
    if k == -3:
      predicted_spans = set(zip(span_starts, span_ends)) & gold_spans
    else:
      if k == -2:
        predicted_starts = pred_starts
        predicted_ends = pred_ends
        if debugging:
          print ("Predicted", list(zip(sorted_starts, sorted_ends, sorted_scores))[:len(gold_spans)])
          print ("Gold", gold_spans)
     # FIXME: scalar index error
      elif k == 0:
        is_predicted = span_scores > 0
        predicted_starts = span_starts[is_predicted]
        predicted_ends = span_ends[is_predicted]
      else:
        if k == -1:
          num_predictions = len(gold_spans)
        else:
          num_predictions = (k * text_length) / 100
        predicted_starts = sorted_starts[:num_predictions]
        predicted_ends = sorted_ends[:num_predictions]
      predicted_spans = set(zip(predicted_starts, predicted_ends))
    evaluator.update(gold_set=gold_spans, predicted_set=predicted_spans)


def _print_f1(total_gold, total_predicted, total_matched, message=""):
  precision = 100.0 * total_matched / total_predicted if total_predicted > 0 else 0
  recall = 100.0 * total_matched / total_gold if total_gold > 0 else 0
  f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
  # print ("{}: Precision: {}, Recall: {}, F1: {}".format(message, precision, recall, f1))
  return precision, recall, f1


def compute_span_f1(gold_data, predictions, task_name):
  assert len(gold_data) == len(predictions)
  total_gold = 0
  total_predicted = 0
  total_matched = 0
  total_unlabeled_matched = 0
  label_confusions = Counter()  # Counter of (gold, pred) label pairs.

  for i in range(len(gold_data)):
    gold = gold_data[i]
    pred = predictions[i]
    total_gold += len(gold)
    total_predicted += len(pred)
    for a0 in gold:
      for a1 in pred:
        if a0[0] == a1[0] and a0[1] == a1[1]:
          total_unlabeled_matched += 1
          label_confusions.update([(a0[2], a1[2]),])
          if a0[2] == a1[2]:
            total_matched += 1
  prec, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, task_name)
  ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched, "Unlabeled " + task_name)
  return prec, recall, f1, ul_prec, ul_recall, ul_f1, label_confusions


def compute_unlabeled_span_f1(gold_data, predictions, task_name):
  assert len(gold_data) == len(predictions)
  total_gold = 0
  total_predicted = 0
  total_matched = 0
  total_unlabeled_matched = 0
  label_confusions = Counter()  # Counter of (gold, pred) label pairs.

  for i in range(len(gold_data)):
    gold = gold_data[i]
    pred = predictions[i]
    total_gold += len(gold)
    total_predicted += len(pred)
    for a0 in gold:
      for a1 in pred:
        if a0[0] == a1[0] and a0[1] == a1[1]:
          total_unlabeled_matched += 1
          label_confusions.update([(a0[2], a1[2]),])
          if a0[2] == a1[2]:
            total_matched += 1
  prec, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, task_name)
  ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched, "Unlabeled " + task_name)
  return prec, recall, f1, ul_prec, ul_recall, ul_f1, label_confusions


def compute_srl_f1(sentences, gold_srl, predictions, srl_conll_eval_path):
  assert len(gold_srl) == len(predictions)  # == number of sentences
  total_gold = 0  # total number of relations (pred->arg arcs) in gold for these sentences
  total_predicted = 0  # total number of predicted relations (pred->arg arcs) for these sentences
  total_matched = 0  # total number of correctly predicted (predicate, argument, role) triples for these sentences
  total_unlabeled_matched = 0  # total number of correctly predicted (predicate, argument) pairs (role might be wrong) for these sentences
  comp_sents = 0  # number of completely correctly predicted sentences
  label_confusions = Counter()

  # Compute unofficial F1 of SRL relations.
  for gold, prediction in zip(gold_srl, predictions):  # gold and predicted SRL for a given sentence
    gold_rels = 0  # number of relations (pred->arg arcs) in gold data for this sentence
    pred_rels = 0  # number of predicted relations (pred->arg arcs) for this sentence
    matched = 0  # number of correctly predicted (predicate, argument, role) triples for this sentence

    for pred_id, gold_args in gold.items():
      filtered_gold_args = [a for a in gold_args if a[2] not in ["V", "C-V"]]
      total_gold += len(filtered_gold_args)
      gold_rels += len(filtered_gold_args)
      if pred_id not in prediction:
        continue
      for a0 in filtered_gold_args:  # gold arguments for this predicate
        for a1 in prediction[pred_id]:  # predicted arguments for this predicate
          if a0[0] == a1[0] and a0[1] == a1[1]:
            # argument spans match
            total_unlabeled_matched += 1
            label_confusions.update([(a0[2], a1[2]),])
            if a0[2] == a1[2]:
              total_matched += 1
              matched += 1
    for pred_id, args in prediction.items():
      filtered_args = [a for a in args if a[2] not in ["V"]] # "C-V"]]
      total_predicted += len(filtered_args)
      pred_rels += len(filtered_args)

    if gold_rels == matched and pred_rels == matched:
      comp_sents += 1

  precision, recall, f1 = _print_f1(total_gold, total_predicted, total_matched, "SRL (unofficial)")
  ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_predicted, total_unlabeled_matched, "Unlabeled SRL (unofficial)")

  # Prepare to evaluate F1 if we want it, need to reconstruct from srl_conll_eval_path
  # print_to_conll(sentences, gold_srl, gold_path, None)
  if not srl_conll_eval_path:
    return precision, recall, f1, -1, -1, -1, ul_prec, ul_recall, ul_f1, None, None
  else:
    # Need to reformat srl_conll_eval_path to the correct extension
    gold_path = srl_conll_eval_path
    gold_predicates = read_gold_predicates(gold_path)
    temp_output = "srl_pred_%d.tmp" % os.getpid()

  print_to_conll(sentences, predictions, temp_output, gold_predicates)

  # Evalute twice with official script.
  if ignore_errors:
      stderr = {"stderr": subprocess.DEVNULL}
  else:
      stderr = {}
  child = subprocess.Popen('sh {} {} {}'.format(
      _SRL_CONLL_EVAL_SCRIPT, gold_path, temp_output), shell=True, stdout=subprocess.PIPE, **stderr)
  eval_info = child.communicate()[0].decode("utf-8")
  child2 = subprocess.Popen('sh {} {} {}'.format(
      _SRL_CONLL_EVAL_SCRIPT, temp_output, gold_path), shell=True, stdout=subprocess.PIPE, **stderr)
  eval_info2 = child2.communicate()[0].decode("utf-8")
  try:
    conll_recall = float(eval_info.strip().split("\n")[6].strip().split()[5])
    conll_precision = float(eval_info2.strip().split("\n")[6].strip().split()[5])
    if conll_recall + conll_precision > 0:
      conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision)
    else:
      conll_f1 = 0
    temp_scores_output = "srl_scores_%d.tmp" % os.getpid()
    temp_scores_file = open(temp_scores_output, 'a+')
    temp_scores_file.write(eval_info)
    temp_scores_file.write(eval_info2)
    temp_scores_file.close()
    print("Official CoNLL Precision={}, Recall={}, Fscore={}. Full: {}".format(
        conll_precision, conll_recall, conll_f1, temp_scores_output))
  except IndexError:
    conll_recall = 0
    conll_precision = 0
    conll_f1 = 0
    # This error message is not usually informative and ends up breaking
    # tqdm, so we will silently fail instead.
    # print("Unable to get FScore. Skipping.")

  return precision, recall, f1, conll_precision, conll_recall, conll_f1, ul_prec, ul_recall, ul_f1, label_confusions, comp_sents


def print_sentence_to_conll(fout, tokens, labels):
  """Print a labeled sentence into CoNLL format.
  """
  for label_column in labels:
    assert len(label_column) == len(tokens)
  for i in range(len(tokens)):
    fout.write(tokens[i].ljust(15))
    for label_column in labels:
      tag = label_column[i]
      if len(tag) >= 15:
        fout.write(tag.rjust(len(tag) + 2))
      else:
        fout.write(tag.rjust(15))
    fout.write("\n")
  fout.write("\n")


def read_gold_predicates(gold_path):
  fin = codecs.open(gold_path, "r", "utf-8")
  gold_predicates = [[],]
  for line in fin:
    line = line.strip()
    if not line:
      gold_predicates.append([])
    else:
      info = line.split()
      gold_predicates[-1].append(info[0])
  fin.close()
  return gold_predicates


def print_to_conll(sentences, srl_labels, output_filename, gold_predicates):
  fout = codecs.open(output_filename, "w", "utf-8")
  for sent_id, words in enumerate(sentences):
    if gold_predicates:
      assert len(gold_predicates[sent_id]) == len(words)
    pred_to_args = srl_labels[sent_id]
    props = ["-" for _ in words]
    col_labels = [["*" for _ in words] for _ in range(len(pred_to_args))]
    for i, pred_id in enumerate(sorted(pred_to_args.keys())):
      # To make sure CoNLL-eval script count matching predicates as correct.
      if gold_predicates and gold_predicates[sent_id][pred_id[0]] != "-":
        props[pred_id[0]] = gold_predicates[sent_id][pred_id[0]]
      else:
        props[pred_id[0]] = "P" + words[pred_id[0]]
      flags = [False for _ in words]
      for start, end, label, _ in pred_to_args[pred_id]:
        # if not max(flags[start:end+1]):
        if True:
          col_labels[i][start] = "(" + label + col_labels[i][start]
          col_labels[i][end] = col_labels[i][end] + ")"
          for j in range(start, end+1):
            flags[j] = True
      # Add unpredicted verb (for predicted SRL).
      if not flags[pred_id[0]]:
        col_labels[i][pred_id[0]] = "(V*)"
    print_sentence_to_conll(fout, props, col_labels)
  fout.close()
