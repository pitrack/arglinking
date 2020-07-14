import numpy as np
import torch
from typing import Optional

from collections import defaultdict, Counter
from operator import itemgetter

DUMMY_POSITION = -999
DUMMY_START = DUMMY_POSITION
DUMMY_END = DUMMY_POSITION
DUMMY_SCORE = 0  # dummy argument gets a hard-coded score of 0 for satisfying a trigger+role
DUMMY_ARG_IDX = -1
_CORE_ROLES = defaultdict(list,  # default_factory
                          conll2012=["ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5", "ARGA"])
DEFAULT_ROLE = defaultdict(str,  # default_factory
                           conll2012='ARG0')

DIAGNOSIS_BEAM_SIZES = range(0,11)

def masked_normalize_confidence(link_scores):
    x = link_scores.clone()
    x[x<0] = 0  # mask out negative values
    x = x/x.sum(-1).unsqueeze(-1).expand_as(x)  # normalize scores to [0, 1] over arguments
    x[torch.isnan(x)] = 0
    return x

# CONFIDENCE = lambda link_scores: torch.nn.functional.softmax(link_scores, dim=-1)  # softmax over arguments
CONFIDENCE = lambda link_scores: masked_normalize_confidence(link_scores)

## Task configurations
CONLL2012 = {
    "granularity": "sentence",
    "include_strings": False,
    "links_may_cross_sentence_boundary": False,
    "trigger_spans_may_cross_sentence_boundary": False,
    "arg_spans_may_cross_sentence_boundary": False,
    "reported_token_offset_granularity": "sentence",
    "spans_may_overlap": False,
}

GVDB = {
    "granularity": "document",
    "include_strings": True,
    "links_may_cross_sentence_boundary": True,
    "trigger_spans_may_cross_sentence_boundary": True,  # trigger span is the entire document, so it (in general) crosses sentence boundaries
    "arg_spans_may_cross_sentence_boundary": True,  # sentence boundaries determined by spaCy, so they may not be completely correct
    "reported_token_offset_granularity": "document",
    "spans_may_overlap": False,
}

RAMS = {
    "granularity": "document",
    "include_strings": False,
    "links_may_cross_sentence_boundary": True,
    "trigger_spans_may_cross_sentence_boundary": False,
    "arg_spans_may_cross_sentence_boundary": False,
    "reported_token_offset_granularity": "document",
    "spans_may_overlap": True,
}

TASK_CONFIGS = {
    "conll2012": CONLL2012,
    "gvdb": GVDB,
    "rams": RAMS,
}


def subselect_gold(num_gold, k, mention_scores, spans, span_embeddings, sentence_ids):
    """
    Randomly pick k of the gold triggers, so that it fits in memory.
    """
    if num_gold > k:
        span_indices = torch.randperm(num_gold, device=mention_scores.device)[:k]
        mention_scores = mention_scores[:, span_indices]
        spans = spans[:, span_indices]
        span_embeddings = span_embeddings[:, span_indices]
        sentence_ids = sentence_ids[:, span_indices]
    else:
        span_indices = None
    return (mention_scores, spans, span_embeddings, span_indices, sentence_ids)

def prune_targets(target_trigger_indices: torch.LongTensor,
                  target_arg_indices: torch.LongTensor,
                  target_roles: torch.LongTensor,
                  trigger_span_indices: Optional[torch.LongTensor],
                  arg_span_indices: Optional[torch.LongTensor],
                  prune_trigger=True,
                  prune_arg=True):
    """
    Objective: Find target triples that are still compatible with
    the trigger and arguments found by the pruner. Also, reindex the indices
    respect to the order determined by the pruner.

    The algorithm is to intersect target_trigger_idx with trigger_span_indices,
    to determine a binary mask of survivors (and to get new indices).
    The same is done for arg.

    target_trigger_indices: Has length num_targets, has values between 0
        and num_trigger_spans where num_trigger_spans is all possible trigger spans.
    target_arg_indices: Has length num_targets, has values between 0
        and num_arg_spans where num_trigger_spans is all possible arg spans.
    target_roles: Has length num_targets
    trigger_span_indices: Has length trigger_k, has values between 0 and
        num_trigger_spans
    arg_span_indices: Has length arg_k, has values between 0 and
        num_arg_spans
    Experimental parameters:
    prune_trigger: Whether to prune trigger
    prune_arg: Whether to prune arg
    """
    # Here, gip stands for "gold in pruned"
    if prune_trigger:
        # both shape (1, num_targets)
        gip_trigger_mask, gip_trigger_indices = _compute_intersection_indices(
            trigger_span_indices, target_trigger_indices)
    else:
        gip_trigger_mask = torch.ones_like(target_trigger_indices).byte()
        gip_trigger_indices = target_trigger_indices

    if prune_arg:
        # both shape (1, num_targets)
        gip_arg_mask, gip_arg_indices = _compute_intersection_indices(
            arg_span_indices, target_arg_indices)
    else:
        gip_arg_mask = torch.ones_like(target_arg_indices).byte()
        gip_arg_indices = target_arg_indices

    # Shape (1, target_arg_indices)
    # Note new_num_targets = sum(combined_masked, dim=-1)
    combined_mask = (gip_trigger_mask * gip_arg_mask).squeeze(0).nonzero().squeeze(-1)
    # This only works on a single document, so we assume batch size 1 again
    # Shape: (1, new_num_targets)
    new_trigger_list = gip_trigger_indices[:, combined_mask]
    new_arg_list = gip_arg_indices[:, combined_mask]
    new_roles = target_roles[:, combined_mask]
    return new_trigger_list, new_arg_list, new_roles


def _compute_intersection_indices(ordered_set_A: torch.LongTensor,
                                 multiset_B: torch.LongTensor):
    """
    General function. Given a set A and a multiset B represented
    by indices into a third set C, compute
    1) A mask elements of B that are in A
    2) If b \\in B is in A, then the index of b in A, 0 otherwise

    In conjunction, this can be used to determine the index of elements
    in A \\cap B with respect to A.

    Note this is useful for pruning and reindexing B w.r.t A
    """

    # Shape: (.., |A|, |B|)
    intersection = (ordered_set_A.unsqueeze(-1) == multiset_B.unsqueeze(-2))
    # Shape: (.., |B|), range [0, 1]
    B_in_A_mask = intersection.any(dim=-2)

    # Shape: (.., |B|), range [0, |A|]
    # Maybe there are floating point issues here?
    B_in_A_indices = torch.arange(intersection.shape[-2],
                                  device=intersection.device).float().matmul(intersection.float()).long()

    return B_in_A_mask, B_in_A_indices

def get_token_distances(trigger_spans, arg_spans):
    """
    Just returns distances as a basic metric
    """
    trigger_starts = trigger_spans[:,:,0].unsqueeze(2)  # (batch_size, num_triggers, 1)
    trigger_ends = trigger_spans[:,:,1].unsqueeze(2)  # (batch_size, num_triggers, 1)
    arg_starts = arg_spans[:,:,0].unsqueeze(1)  # (batch_size, 1, num_args)
    arg_ends = arg_spans[:,:,1].unsqueeze(1)  # (batch_size, 1, num_args)

    # token_distances[batch][i][j] = number of tokens between trigger `i` and argument `j`
    # It also returns a matching tensor; 0 if trigger is after arg, 1 if trigger is before arg
    # Shape: (batch_size, num_triggers, num_args) [broadcasted]
    token_distances, _ = torch.stack([trigger_starts - arg_ends, arg_starts - trigger_ends],
                                     dim=-1).max(dim=-1)
    return token_distances

def trigger_arg_in_same_sentence(trigger_sent_ids, arg_sent_ids):
    trigger_sent_ids = trigger_sent_ids.unsqueeze(-1)  # (batch_size, num_triggers, 1)
    arg_sent_ids = arg_sent_ids.unsqueeze(-2)  # (batch_size, 1, num_args)
    same_sentence_mask = torch.eq(trigger_sent_ids, arg_sent_ids)
    return same_sentence_mask

def autopopulate_params(vocab, params):
    """
    Compute/autopopulate input dimensionalities
    Everything here should be a local variable, stored in params, or explicitly returned
    """

    # Compute word embs
    token_embs = params['token_embedders']
    word_dim = token_embs['tokens']['embedding_dim']
    charcnn_params = token_embs['token_characters']['encoder']
    char_emb_dim = len(charcnn_params['ngram_filter_sizes']) * charcnn_params['num_filters']
    contextless_word_emb_dim = word_dim + char_emb_dim
    params['context_layer']['input_size'] = params['lm_size'] + contextless_word_emb_dim

    # Compute span embedding dims
    context_hidden_dim = params['context_layer']['hidden_size']
    context_multiplier = 2 if params['context_layer']['bidirectional'] else 1
    span_embedding_dim = contextless_word_emb_dim + 2 * context_multiplier * context_hidden_dim # account for attn and two endpoints

    # Compute input dims
    feature_dim = params['feature_size']
    role_dim = params['role_dim']
    trigger_dim = span_embedding_dim + feature_dim
    arg_dim = span_embedding_dim + feature_dim
    trigger_role_dim = trigger_dim + role_dim
    params['trigger_mention_feedforward']['input_dim'] = trigger_dim
    params['arg_mention_feedforward']['input_dim'] = arg_dim
    params['trigger_role_feedforward']['input_dim'] = trigger_role_dim
    params['trigger_role_projection']['input_dim'] = trigger_role_dim
    params['trigger_role_projection']['hidden_dims'] = params['arg_mention_feedforward']['input_dim']

    params['arg_role_feedforward']['input_dim'] = arg_dim + role_dim

    params['distance_feedforward']['input_dim'] = feature_dim

    # Compute big span embedding size
    num_link_features = sum(params['arch_params']['span_pair_feats'].values())
    params['link_feedforward']['input_dim'] = 3 * params['arg_mention_feedforward']['input_dim'] + num_link_features * feature_dim

    num_roles = vocab.get_vocab_size("labels")
    return span_embedding_dim, trigger_dim, arg_dim, num_roles

def analyze_beam(output_dict):
    """
    Compare beam arguments to gold arguments.
    For each trigger: set(gold_args) - set(beam_args)
    """
    link_scores = output_dict["link_scores"].detach().cpu()
    sort_map = output_dict["candidate_arg_indices"].detach().cpu()
    triggers = output_dict["trigger_spans"].detach().cpu()
    arguments = output_dict["arg_spans"].detach().cpu()

    trigger_spans = triggers.tolist()
    argument_spans = arguments.tolist()

    metadata = output_dict["metadata"][0]
    gold_triggers = metadata["triggers"]
    gold_arguments = metadata["arguments"]

    missed_args_counters = {k: Counter() for k in DIAGNOSIS_BEAM_SIZES}

    sort_map_list = sort_map.tolist()
    doc_i = 0
    for trigger_idx, trigger_span in enumerate(trigger_spans[doc_i]):
      trigger_span = tuple(trigger_span)
      beam_arg_indices = sort_map_list[trigger_idx]
      beam_arg_spans = [tuple(argument_spans[doc_i][bai]) for bai in beam_arg_indices]

      indices = [i for i, x in enumerate(gold_triggers) if x == trigger_span]

      gold_arg_spans = [gold_arguments[i] for i in indices]
      for beam_size in DIAGNOSIS_BEAM_SIZES:
          num_missed_args = len(set(gold_arg_spans) - set(beam_arg_spans[:beam_size]))
          missed_args_counters[beam_size].update([num_missed_args])

    return missed_args_counters

def _combine_args(task, triggers, args, roles, confs, sentence_lengths, sentence_start_offsets, strings):
    """
    Combine together information per sentence and per trigger
    """
    task_config = TASK_CONFIGS[task]
    assert len(sentence_lengths) == len(sentence_start_offsets)

    if task_config["granularity"] == "sentence":
        sentence_triggers = [{} for _ in sentence_lengths]
        if task_config["include_strings"]:
            sentence_trigger_strings = [{} for _ in sentence_lengths]
        else:
            sentence_trigger_strings = None
    elif task_config["granularity"] == "document":
        sentence_triggers = [{}]
        if task_config["include_strings"]:
            sentence_trigger_strings = [{}]
        else:
            sentence_trigger_strings = None
    else:
        raise NotImplementedError

    span_to_sent_id_map = np.repeat(np.arange(len(sentence_lengths)),
                                    sentence_lengths)

    data = [triggers, args, roles, confs, strings]

    for trigger, arg, role, conf, string in zip(*data):
        trigger_crosses_sentence_boundary = (span_to_sent_id_map[trigger[0]] != span_to_sent_id_map[trigger[1]])
        arg_crosses_sentence_boundary = (span_to_sent_id_map[arg[0]] != span_to_sent_id_map[arg[1]])
        link_crosses_sentence_boundary = (span_to_sent_id_map[trigger[0]] != span_to_sent_id_map[arg[0]])
        if trigger_crosses_sentence_boundary and not task_config["trigger_spans_may_cross_sentence_boundary"]:
            continue
        if arg_crosses_sentence_boundary and not task_config["arg_spans_may_cross_sentence_boundary"]:
            continue
        if link_crosses_sentence_boundary and not task_config["links_may_cross_sentence_boundary"]:
            continue

        sent_id = span_to_sent_id_map[trigger[0]]
        # Re-adjust token offsets if necessary.
        # Model uses document-level offsets, but certain tasks may use other kinds of offset.
        if task_config["reported_token_offset_granularity"] == "sentence":
            start_offset = sentence_start_offsets[sent_id]
        elif task_config["reported_token_offset_granularity"] == "document":
            start_offset = 0
        else:
            raise ValueError(f"unrecognized reported_token_offset_granularity: {task_config['reported_token_offset_granularity']}")
        trigger = (trigger[0] - start_offset, trigger[1] - start_offset)
        arg = (arg[0] - start_offset, arg[1] - start_offset)

        labeled_span = (arg[0], arg[1], role, conf)
        string_span = (role, string)

        # Remap sentence id for accessing data structures
        if task_config["granularity"] == "document":
            sent_id = 0
        if trigger in sentence_triggers[sent_id]:
            sentence_triggers[sent_id][tuple(trigger)].append(labeled_span)
            if task_config["include_strings"]:
                sentence_trigger_strings[sent_id][tuple(trigger)].append(string_span)
        else:
            sentence_triggers[sent_id][tuple(trigger)] = [labeled_span]
            if task_config["include_strings"]:
                sentence_trigger_strings[sent_id][tuple(trigger)] = [string_span]

    output = dict(structures=sentence_triggers,
                  strings=sentence_trigger_strings)
    return output

def _interpret_preds(task, trigger_spans, argument_spans, sort_map, t_idx, a_idx, r, confidences, label_vocab, i, text):
    trigger_span = tuple(trigger_spans[i][t_idx])
    corrected_arg_idx = sort_map[t_idx][a_idx]
    arg_span = tuple(argument_spans[i][corrected_arg_idx])
    role_val = label_vocab[r] if r in label_vocab else DEFAULT_ROLE[task]
    pred_string = " ".join(text[arg_span[0]:arg_span[1]+1])
    conf = confidences[i][t_idx][r][a_idx+1].item()  # offset arg index by 1 to account for dummy arg

    return (trigger_span, arg_span, role_val, conf, pred_string)

def argmax_decode(output_dict, text):
    """
    Fills each trigger+role slot with the highest link-scoring argument (incl. dummy).
    Each slot is filled with exactly one argument.
    """
    # arguments[sort_map[i]] is the i-th nearest argument span to the trigger
    link_scores = output_dict["link_scores"].detach().cpu()
    sort_map = output_dict["candidate_arg_indices"].detach().cpu()
    triggers = output_dict["trigger_spans"].detach().cpu()
    arguments = output_dict["arg_spans"].detach().cpu()
    vocab = output_dict["vocab"]
    metadata = output_dict["metadata"]
    task = output_dict["task"]

    label_vocab = vocab.get_index_to_token_vocabulary("labels")

    if len(metadata) != 1:
        raise ValueError("expected metadata to be for a single document")

    _, predicted_arguments = link_scores.max(-1)

    # Subtract one here because index 0 is the dummy "no argument" class,
    # so this makes the indices line up with actual spans if the prediction
    # is greater than -1.
    predicted_arguments -= 1

    trigger_spans = triggers.tolist()
    argument_spans = arguments.tolist()

    confidences = CONFIDENCE(link_scores)

    for i, metadata in enumerate(metadata):
        pred_args_idx = predicted_arguments[i].view(-1).tolist()
        (num_preds, num_roles) = predicted_arguments[i].shape
        pred_idx = np.repeat(np.arange(num_preds), num_roles)
        roles = np.tile(np.arange(num_roles), num_preds)

        merged_predictions = [_interpret_preds(task, trigger_spans, argument_spans, sort_map, t_idx, a_idx, r, confidences, label_vocab, i, text)
                              for t_idx, a_idx, r in zip(pred_idx, pred_args_idx, roles)
                              if a_idx >= 0]

        return merged_predictions

def _extract_tuples(link_scores, triggers, arguments, sort_map):
    trigger_spans = triggers.tolist()
    argument_spans = arguments.tolist()
    sort_map = sort_map.tolist()

    confidences = CONFIDENCE(link_scores)

    link_scores = link_scores.tolist()  # [num_triggers, num_roles, num_candidate_args]

    tuples = []  # elements are (trigger_start, trigger_end, arg_start, arg_end, role, score, trigger_idx, arg_idx)
    for i,trigger_span in enumerate(trigger_spans):
        candidate_argument_spans = arguments[sort_map[i]].tolist()  # arguments that this trigger may be in a relation with
        candidate_argument_spans = [[DUMMY_START, DUMMY_END]] + candidate_argument_spans  # prepend dummy argument span
        for j,candidate_argument_span in enumerate(candidate_argument_spans):
            for role_idx,_ in enumerate(link_scores[i]):
                score = link_scores[i][role_idx][j]
                conf = confidences[i][role_idx][j].item()

                # filter out candidate argument spans that score less than the dummy argument span
                if score > DUMMY_SCORE:
                    assert j != 0  # dummy should not be included in list of explicit predictions

                    # use `j`-1 as the argument index into `candidate_argument_spans` for this trigger; shifted due to presence of dummy arg in first position of `candidate_argument_spans`
                    t = (trigger_span[0], trigger_span[1], candidate_argument_span[0], candidate_argument_span[1], role_idx, conf, i, j-1)
                    tuples.append(t)

    return tuples

def _contains(a, b, c, d):
    """
    Whether inclusive interval [a,b] contains interval [c,d]
    """
    if not ((a <= b) and (c <= d)):
        raise ValueError("Left endpoint must be given before right endpoint: [{}, {}] does not contain [{}, {}]".format(a,b,c,d))
    return (a <= c) and (b >= d)

def _no_overlap_with_trigger(t):
    trigger_start, trigger_end, arg_start, arg_end, _, _, _, _= t
    return not _contains(arg_start, arg_end, trigger_start, trigger_end)

def _filter_tuples(tuples):
    """
    https://arxiv.org/abs/1810.02245

    Filter out tuples:
        (i) whose boundary (i, j) overlaps with the trigger position (p, p)
       (ii) whose score is lower than that of the dummy (done implicitly when constructing tuple list in `_extract_tuples()`)
    """
    tuples = list(filter(_no_overlap_with_trigger, tuples))
    return tuples

def _is_overlap(i, j, used_spans):
    """
    Returns whether the span (i, j) overlaps with the interval(s) described by the items in `used_spans`
    """
    minval = min([i] + [s[0] for s in used_spans])  # smallest start index among spans
    maxval = max([j] + [s[1] for s in used_spans])  # largest end index among spans
    used_words = [0]*(maxval - minval + 1)
    for (span_start, span_end) in used_spans:
        assert used_words[span_start-minval: span_end-minval+1] == [0]*(span_end-span_start+1)  # check that no span has covered these words, which should be the case because spans aren't allowed to overlap
        used_words[span_start-minval: span_end-minval+1] = [1]*(span_end-span_start+1)  # (inclusive) interval has now been covered by a span

    return sum(used_words[i-minval: j-minval+1]) > 0

def _ouchi_greedy_decode(link_scores, arguments, sort_map, triggers, core_roles, span_consistent):
    # ASSUMPTION: function arguments correspond to predictions for a single document
    # Shape: [num_triggers, num_roles, num_candidate_args]
    link_scores = link_scores.squeeze(0)
    num_triggers, num_roles, num_candidate_args = link_scores.shape

    # Shape: [num_args, 2]
    arguments = arguments.squeeze(0)

    # Shape: [num_triggers, 2]
    triggers = triggers.squeeze(0)

    spans = defaultdict(list)
    used_cores = defaultdict(list)  # core roles we've used for a given trigger
    tuples = _extract_tuples(link_scores, triggers, arguments, sort_map)  # elements are (trigger_start, trigger_end, arg_start, arg_end, role, score)
    tuples = _filter_tuples(tuples)

    # Sort spans in decreasing order by score
    sorted_tuples = sorted(tuples, key=itemgetter(5), reverse=True)

    # for a given trigger index i: [(argument_start, argument_end), ...] = arguments[sort_map[i], :]
    used_spans = defaultdict(list)  # spans we've covered for a given trigger
    for (trigger_start, trigger_end, arg_start, arg_end, role, conf, trigger_idx, arg_idx) in sorted_tuples:
        if (role not in used_cores[trigger_idx]) and not (span_consistent and _is_overlap(arg_start, arg_end, used_spans[trigger_idx])):
            spans[trigger_idx].append((trigger_start, trigger_end, arg_start, arg_end, role, conf, trigger_idx, arg_idx))
            used_spans[trigger_idx].append((arg_start, arg_end))
            if role in core_roles:
                used_cores[trigger_idx].append(role)

    return spans

def greedy_decode(output_dict, text):
    """
    Greedy decoding algorithm from: "A Span Selection Model for Semantic Role Labeling" (Ouchi, et al., 2018)
    <https://arxiv.org/abs/1810.02245>
    (Algorithm 1, Appendix A)

    Each slot is filled with at least one (potentially dummy) argument.
    """
    # arguments[sort_map[i]] is the i-th nearest argument span to the trigger
    link_scores = output_dict["link_scores"].detach().cpu()
    sort_map = output_dict["candidate_arg_indices"].detach().cpu()
    triggers = output_dict["trigger_spans"].detach().cpu()
    arguments = output_dict["arg_spans"].detach().cpu()
    vocab = output_dict["vocab"]
    metadata = output_dict["metadata"]
    task = output_dict["task"]
    span_consistent = not TASK_CONFIGS[task]["spans_may_overlap"]

    if len(metadata) != 1:
        raise ValueError("expected metadata to be for a single document")

    label_vocab = vocab.get_index_to_token_vocabulary("labels")
    inv_srl_vocab = {v: k for k, v in label_vocab.items()}
    core_roles = [inv_srl_vocab[r] for r in _CORE_ROLES[task]]

    predictions = _ouchi_greedy_decode(link_scores, arguments, sort_map, triggers, core_roles, span_consistent)

    trigger_spans = triggers.tolist()
    argument_spans = arguments.tolist()

    confidences = CONFIDENCE(link_scores)

    for i, metadata in enumerate(metadata):
        merged_predictions = []
        for trigger_idx, prediction_list in predictions.items():
            for prediction in prediction_list:
                _, _, _, _, role, conf, trigger_idx, arg_idx = prediction
                assert arg_idx >= 0  # dummy arguments aren't explicitly predicted
                merged_predictions.append(_interpret_preds(task, trigger_spans, argument_spans, sort_map, trigger_idx, arg_idx, role, confidences, label_vocab, i, text=text))

        return merged_predictions
