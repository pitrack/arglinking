import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn import Parameter
from overrides import overrides

import h5py
import json
import os
import shutil

import miso.data.vocabulary as Vocabulary
from miso.models.model import Model
from miso.modules.token_embedders import Embedding, TokenCharactersEncoder
from miso.modules import FeedForward, StackedBidirectionalLstm, TimeDistributed, ScalarMix
from miso.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from miso.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from miso.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from miso.utils import nn as util
from miso.utils.contextual_embedding_cache import load_lm_embeddings
import miso.utils.argl as al_util
from miso.utils.initializers import InitializerApplicator
from miso.utils.environment import move_to_device
from miso.metrics import ConllSrlScores, GVDBScores


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

GRANULARITIES = ["sentence", "document"]
DECODERS = ["argmax", "greedy"]
SCORERS = {"ConllSrlScores": ConllSrlScores,
           "GVDBScores": GVDBScores}

class ArgLinkingModel(Model):
    """
    This ``Model`` performs Argument Linking by introducing latent argument vectors for each
    of a trigger's role slots, then linking the latent argument vectors with
    argument spans from the text.

    `partner` denotes the kind of span being linked to.
    For argument linking, `partner` refers to an argument.
    """
    def __init__(self,
                 task: str,
                 vocab: Vocabulary,
                 dev_gold_data_path: str,
                 test_gold_data_path: str,
                 required_components: Dict[str, bool],
                 text_field_embedder: TextFieldEmbedder,
                 context_layer: PytorchSeq2SeqWrapper,
                 trigger_mention_feedforward: FeedForward,
                 arg_mention_feedforward: FeedForward,
                 trigger_role_feedforward: FeedForward,
                 trigger_role_projection: FeedForward,
                 arg_role_feedforward: FeedForward,
                 link_feedforward: FeedForward,
                 distance_feedforward: FeedForward,
                 role_embeddings: Parameter,
                 num_widths: int,
                 max_args_per_trigger: int,
                 coarse_scorer: Parameter,
                 num_genres: int,
                 trigger_spans_per_word: float,
                 arg_spans_per_word: float,
                 max_training_doc_size: int, # where to clip trigger/arg per word
                 max_inference_doc_size: int,
                 feature_size: int,
                 lm_loaders,
                 lm_layers: List[int],
                 arch_params: dict,
                 empty_cache: bool,
                 default_dropout: float,
                 lexical_dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer = None) -> None:
        super(ArgLinkingModel, self).__init__()

        self._task = task

        self._vocab = vocab
        self._dev_gold_data_path = dev_gold_data_path
        self._test_gold_data_path = test_gold_data_path

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer

        # Note that if output_dim of the Linear layer is self._max_roles,
        # we are learning a separate scorer for each role, which means we can't use the
        # role information to calculate this later
        self._trigger_role_projection = TimeDistributed(trigger_role_projection)
        self._trigger_feedforward_scorer = torch.nn.Sequential(
                TimeDistributed(trigger_mention_feedforward),
                TimeDistributed(torch.nn.Linear(trigger_mention_feedforward.get_output_dim(), 1))
        )

        if required_components["require_trigger_role_feedforward_scorer"]:
            self._trigger_role_feedforward_scorer = torch.nn.Sequential(
                    TimeDistributed(trigger_role_feedforward),
                    TimeDistributed(torch.nn.Linear(trigger_role_feedforward.get_output_dim(), 1))
            )
        else:
            self._trigger_role_feedforward_scorer = None

        self._arg_feedforward_scorer = torch.nn.Sequential(
                TimeDistributed(arg_mention_feedforward),
                TimeDistributed(torch.nn.Linear(arg_mention_feedforward.get_output_dim(), 1))
        )

        if required_components["require_arg_role_feedforward_scorer"]:
            self._arg_role_feedforward_scorer = torch.nn.Sequential(
                    TimeDistributed(arg_role_feedforward),
                    TimeDistributed(torch.nn.Linear(arg_role_feedforward.get_output_dim(), 1))
            )
        else:
            self._arg_role_feedforward_scorer = None

        if required_components["require_link_scorer"]:
            self._link_scorer = torch.nn.Sequential(
                    TimeDistributed(link_feedforward),
                    TimeDistributed(torch.nn.Linear(link_feedforward.get_output_dim(), 1))
            )
        else:
            self._link_scorer = None

        if required_components["require_distance_scorer"]:
            self._distance_feedforward_scorer = torch.nn.Sequential(
                    TimeDistributed(distance_feedforward),
                    TimeDistributed(torch.nn.Linear(distance_feedforward.get_output_dim(), 1))
            )
        else:
            self._distance_feedforward_scorer = None

        self._coarse_scorer = coarse_scorer

        self._num_widths = num_widths
        self._endpoint_trigger_span_extractor = EndpointSpanExtractor(context_layer.get_output_dim(),
                                                                      combination="x,y",
                                                                      num_width_embeddings=self._num_widths,
                                                                      span_width_embedding_dim=feature_size,
                                                                      bucket_widths=True)
        self._endpoint_arg_span_extractor = EndpointSpanExtractor(context_layer.get_output_dim(),
                                                                  combination="x,y",
                                                                  num_width_embeddings=self._num_widths,
                                                                  span_width_embedding_dim=feature_size,
                                                                  bucket_widths=True)

        head_part_dim = text_field_embedder.get_output_dim()
        self._attentive_trigger_span_extractor = SelfAttentiveSpanExtractor(input_dim=head_part_dim)
        self._attentive_arg_span_extractor = SelfAttentiveSpanExtractor(input_dim=head_part_dim)
        self._role_embeddings = role_embeddings

        self._max_args_per_trigger = max_args_per_trigger  # each trigger can have up to `_max_args_per_trigger` arguments
        self._trigger_spans_per_word = trigger_spans_per_word
        self._arg_spans_per_word = arg_spans_per_word
        self._max_training_doc_size = max_training_doc_size  # pretend the document has at most this many tokens
        self._max_inference_doc_size = max_inference_doc_size  # max limit during inference

        self._num_distance_buckets = 10  # 10 possible distance buckets.
        self._distance_embedding = Embedding(self._num_distance_buckets, feature_size)
        self._genre_embedding = Embedding(num_genres, feature_size)

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x

        self._default_dropout = default_dropout

        # elmo/bert stuff
        self.single_lm_loader = lambda doc, mode: lm_loaders[mode](doc)
        self._contextualized_embeddings_scalars = ScalarMix(mixture_size=len(lm_layers))  # leave inits default

        self._scorer = SCORERS[arch_params['scorer']](self._vocab)
        self._loss = torch.nn.CrossEntropyLoss(reduction='sum')

        ## Load architecture parameters
        # whether to constrain arguments to be in same sentence as trigger ("sentence")
        # or to allow arguments to be in other sentences ("document")
        self._granularity_level = arch_params['granularity_level']
        self._decoder = arch_params['decoder']
        self._trigger_arg_components = arch_params['trigger_arg_components']
        self._span_pair_feats = arch_params['span_pair_feats']
        self._link_scores_params = arch_params['link_scores']
        self._empty_cache = empty_cache

        if self._granularity_level not in GRANULARITIES:
            raise ValueError(f"unrecognized granularity level: {self._granularity_level}")
        if self._decoder not in DECODERS:
            raise ValueError(f"unrecognized decoder: {self._decoder}")

        self._use_gold_triggers = arch_params['use_gold_triggers']
        self._use_gold_arguments = arch_params['use_gold_arguments']

        self._required_components = required_components

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                source: Tuple[Dict[str, torch.LongTensor],
                              torch.IntTensor,
                              Optional[torch.IntTensor],
                              Optional[List[Dict[str, Any]]]],
                for_training=False,
                for_prediction=False,
                outputdir=None,
                cpu_eval=False) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Arglinking model.
        """
        # Move self and source to cpu.
        if not for_training:
            original_device, source = self._device_move(source, cpu_eval)
        curr_device = source['genre'].device

        self._for_prediction = for_prediction

        # Unpacking for creating contextual embeddings
        tokens, genre, metadata = source["text"], source["genre"], source["metadata"]

        # Create embeddings and genre_emb
        contextualized_embeddings, text_embeddings, genre_emb, document_length = self._load_contextualized_embs(tokens, genre, metadata, for_training)

        # Start processing spans.
        # If we have gold data, `trigger_spans` is just the gold triggers. If we don't have gold data, `trigger_spans` is all possible enumerated spans.
        # (Similarly for `arg_spans`). Items in `trigger_spans` and `arg_spans` are unique, items in `target_*` are not necessarily unique.
        # The elements of `target_trigger_idx` index into `trigger_spans` (similarly for `target_arg_idx` and `arg_spans`).
        (trigger_spans, arg_spans, trigger_sentence_ids, arg_sentence_ids) = (source["all_triggers"], source["all_args"],
                                                                              source["all_trigger_sentence_ids"], source["all_arg_sentence_ids"])
        if for_training:
            (target_role_labels, target_trigger_idx, target_arg_idx) = (source["target_roles"], source["target_trigger_idx"].squeeze(-1), source["target_arg_idx"].squeeze(-1))
        # Compute attentive embeddings -- these will be really big
        # (batch_size, num_{trigger/arg}_spans, embedding_size)
        trigger_span_embeddings = torch.cat([
            self._attentive_trigger_span_extractor(text_embeddings, trigger_spans), # don't want ELMo/BERT here
            self._endpoint_trigger_span_extractor(contextualized_embeddings, trigger_spans)
        ], dim=-1)
        #trigger_span_embeddings = torch.zeros_like(trigger_span_embeddings)  # baseline: no/fixed trigger representation

        arg_span_embeddings = torch.cat([
            self._attentive_arg_span_extractor(text_embeddings, arg_spans), # don't want ELMo/BERT here
            self._endpoint_arg_span_extractor(contextualized_embeddings, arg_spans)
        ], dim=-1)

        # Individal mention scores
        # Limit number of predictions for memory during training; no memory problems on CPU
        if for_training:
            effective_doc_size = min(document_length, self._max_training_doc_size)
        elif arg_span_embeddings.is_cuda: # need to fit
            effective_doc_size = min(document_length, self._max_inference_doc_size)
        else:
            effective_doc_size = document_length # cpu, inference. Must want full doc

        trigger_k = int(effective_doc_size * self._trigger_spans_per_word)
        trigger_k = max(trigger_k, 1)
        arg_k = int(effective_doc_size * self._arg_spans_per_word)
        arg_k = min(max(arg_k, 2), arg_spans.size(1))  # 2 <= arg_k <= number of argument spans

        # Compute scores analogous to P(e) and P(v).
        trigger_mention_scores = self._trigger_feedforward_scorer(trigger_span_embeddings).squeeze(-1)  # how good of a trigger span this is
        arg_mention_scores = self._arg_feedforward_scorer(arg_span_embeddings).squeeze(-1)  # how good of an argument span this is

        # Either sample a subset of gold (to fit memory), or prune and remap
        if self._use_gold_triggers:
            (trigger_mention_scores, trigger_spans, trigger_span_embeddings,
             trigger_span_indices, trigger_sentence_ids) = al_util.subselect_gold(trigger_spans.size(1),
                                                                                  trigger_k,
                                                                                  trigger_mention_scores,
                                                                                  trigger_spans,
                                                                                  trigger_span_embeddings,
                                                                                  trigger_sentence_ids)
        else:
            trigger_mention_scores, trigger_span_indices = torch.topk(trigger_mention_scores, trigger_k, dim=-1)
            # makes batch_size = 1 assumption at this squeeze
            trigger_spans = trigger_spans[:, trigger_span_indices.squeeze(0), :]
            trigger_span_embeddings = trigger_span_embeddings[:, trigger_span_indices.squeeze(0), :]
            trigger_sentence_ids = trigger_sentence_ids[:, trigger_span_indices.squeeze(0)]

        if self._use_gold_arguments:
            (arg_mention_scores, arg_spans, arg_span_embeddings,
             arg_span_indices, arg_sentence_ids) = al_util.subselect_gold(arg_spans.size(1),
                                                                          arg_k,
                                                                          arg_mention_scores,
                                                                          arg_spans,
                                                                          arg_span_embeddings,
                                                                          arg_sentence_ids)
        else:
            arg_mention_scores, arg_span_indices = torch.topk(arg_mention_scores, arg_k, dim=-1)
            arg_spans = arg_spans[:, arg_span_indices.squeeze(0), :]
            arg_span_embeddings = arg_span_embeddings[:, arg_span_indices.squeeze(0), :]
            arg_sentence_ids = arg_sentence_ids[:, arg_span_indices.squeeze(0)]

        if for_training:
            # target handling. Note: `target_*` is used only to compute loss
            (pruned_target_trigger_indices,
             pruned_target_arg_indices,
             pruned_target_roles) = al_util.prune_targets(target_trigger_idx, target_arg_idx, target_role_labels,
                                                          trigger_span_indices, arg_span_indices,
                                                          prune_trigger=(trigger_span_indices is not None),
                                                          prune_arg=(arg_span_indices is not None))

        if self._granularity_level == "sentence":
            same_sentence_mask = al_util.trigger_arg_in_same_sentence(trigger_sentence_ids, arg_sentence_ids).float().log()  # when finding "best" spans, `1` is better than `0`
        else:
            same_sentence_mask = None

        # Collect dimensional statistics
        num_trigger_spans = trigger_spans.size(1)  # int
        num_arg_spans = arg_spans.size(1)  # int
        num_roles = self._role_embeddings.size(0)  # int
        max_args_per_trigger = min(num_arg_spans, self._max_args_per_trigger)  # number of total arguments may be less than `self._max_args_per_trigger`

        # Handle trigger_role combination
        # Shape: (batch_size, num_triggers, num_roles, trigger_embedding_size + role_embedding_size)
        triggerrole_embeddings_concat = torch.cat([trigger_span_embeddings.unsqueeze(2).repeat(1,1,num_roles,1),
                                                   self._role_embeddings.unsqueeze(0).unsqueeze(1).repeat(1,num_trigger_spans,1,1)],
                                                  dim=-1)
        # [trigger_span_emb; role_emb] -> projection into "entity/argument space"
        # Shape: (batch_size, num_triggers, num_roles, embedding_size)
        triggerrole_embeddings = self._trigger_role_projection(triggerrole_embeddings_concat)

        # Provides some sense of how well the trigger goes with the role
        # Shape: (batch_size, num_triggers, num_roles)
        if self._required_components["require_triggerrole_scores"]:
            triggerrole_scores = self._trigger_role_feedforward_scorer(triggerrole_embeddings_concat).squeeze(-1)
        else:
            triggerrole_scores = None

        # This is in the forward() function because it uses various precomputed components.
        # Uses some broadcasting.
        # Shape: (batch_size, num_triggers, num_args)
        trigger_arg_scores = torch.zeros(1, num_trigger_spans, num_arg_spans, device=curr_device)
        token_distances = al_util.get_token_distances(trigger_spans, arg_spans)
        # Shape: (batch_size, num_triggers, args_per_trigger, embedding_size)
        token_distance_embeddings = self._distance_embedding(
            util.bucket_values(token_distances,
                               num_total_buckets=self._num_distance_buckets))

        if self._trigger_arg_components['coarse_score']:
            # Shape: (batch_size, num_triggers, num_args)
            trigger_arg_scores += self._get_coarse_scores(trigger_span_embeddings, arg_span_embeddings)
        if self._trigger_arg_components['arg_score']:
            # Shape: (batch_size, 1, num_args)
            trigger_arg_scores += arg_mention_scores.unsqueeze(1)
        if self._trigger_arg_components['token_distances']:
            # Shape: (batch_size, num_triggers, num_args)
            token_distance_scores = self._distance_feedforward_scorer(token_distance_embeddings).squeeze(-1)
            trigger_arg_scores += token_distance_scores.float()
        if self._granularity_level == "sentence":
            # give all out-of-sentence arguments a very low score so we look only at arguments in the same sentence as the trigger
            # Shape: (batch_size, num_triggers, num_args)
            trigger_arg_scores += same_sentence_mask  # when finding "best" spans, `1` is better than `0`

        # Prunes down to max_args_per_trigger arguments for each trigger using trigger_arg_scores
        # Returns (batch_size, num_triggers, max_args_per_trigger) or (batch_size, num_trigger, num_args) depending
        candidate_trigger_arg_scores, candidate_arg_indices, inverse_sort_arg_indices = self._find_top_args_for_trigger(trigger_arg_scores, max_args_per_trigger)

        # Shape: (batch_size, num_triggers, max_args_per_trigger, embedding_size)
        candidate_arg_span_embeddings = util.flattened_index_select(arg_span_embeddings,
                                                                    candidate_arg_indices)
        # Shape: (batch_size, num_triggers, max_args_per_trigger, 1)
        candidate_arg_mention_scores = util.flattened_index_select(arg_mention_scores,
                                                                   candidate_arg_indices)

        argrole_scores = None
        if self._link_scores_params['use_argrole_scores']:
            # Save space if not used
            # Shape: (batch_size, num_triggers, num_roles, max_args_per_trigger, arg_embedding_size + role_embedding_size)
            argrole_embeddings_concat = torch.cat([candidate_arg_span_embeddings.unsqueeze(2).repeat(1,1,num_roles,1,1),
                                                   self._role_embeddings.unsqueeze(0).unsqueeze(1).unsqueeze(3).repeat(1,num_trigger_spans,1,max_args_per_trigger,1)],
                                                   dim=-1)
            # Shape: (batch_size, num_triggers, num_roles, max_args_per_trigger)
            argrole_scores = self._arg_role_feedforward_scorer(argrole_embeddings_concat).squeeze(-1)

        # Shape: (batch_size, num_triggers, max_args_per_trigger)
        # ASSUMPTION: batch_size == 1
        span_pair_embeddings = None
        if self._link_scores_params['use_partner_scores']:
            # Save space if not used
            # Each trigger has its own row in `candidate_arg_indices`, so we have to index into the given row in `token_distance_embeddings` appropriately
            candidate_token_distance_embeddings = torch.diagonal(util.flattened_index_select(token_distance_embeddings.squeeze(0),candidate_arg_indices), offset=0, dim1=0, dim2=1).permute(2,0,1).unsqueeze(0)

            # Compute embeddings pairing a trigger+role with an argument (with additional features).
            # Shape: (batch_size, num_triggers, num_roles, max_args_per_trigger, embedding_size)
            span_pair_embeddings = self._compute_span_pair_embeddings(triggerrole_embeddings,
                                                                      candidate_arg_span_embeddings,
                                                                      candidate_token_distance_embeddings,
                                                                      genre_emb)

        # Compute scores for how well an argument satisfies a trigger's role (i.e., how well they "link").
        # Shape: (batch_size, num_triggers, num_roles, max_args_per_trigger + 1), "+ 1" is for dummy argument (no link)
        link_scores = self._compute_link_scores(num_trigger_spans,
                                                max_args_per_trigger,
                                                num_roles,
                                                span_pair_embeddings,
                                                triggerrole_scores,
                                                argrole_scores,
                                                trigger_mention_scores,
                                                candidate_arg_mention_scores,
                                                candidate_trigger_arg_scores  # capture coarse scores (including arg span scores) and same-sentence mask information
                                                )

        output_dict = {
            "link_scores": link_scores.detach().cpu(),
            "candidate_arg_indices": candidate_arg_indices.detach().cpu(),
            "candidate_trigger_arg_scores": candidate_trigger_arg_scores.detach().cpu(),
            "trigger_spans": trigger_spans.detach().cpu(),
            "arg_spans": arg_spans.detach().cpu(),
            "metadata": metadata,
            "decoder": self._decoder,
            "vocab": self._vocab,
            "task": self._task,
        }

        if for_training and metadata[0]["has_gold_targets"]:
            # Find the gold labels (triggers/roles) for the spans which we kept.
            # (batch_size, num_roles)
            gold_labels = pruned_target_roles

            # Reshape (batch_size, num_triggers, num_roles, max_args_per_trigger + 1) -> (batch_size * num_triggers * num_roles, max_args_per_trigger + 1) for `scatter_` to work properly
            shape = [-1, link_scores.size(3)]  # [-1, max_args_per_trigger + 1]
            link_scores = link_scores.view(*shape)  # (batch_size * num_triggers * num_roles, max_args_per_trigger + 1)

            # Recover gold trigger spans and argument spans from indices
            triggerrole_indices = num_roles * pruned_target_trigger_indices + gold_labels # (batch_size, num_triggerarg_arcs)
            gold_arg_indices = torch.zeros(1, num_trigger_spans * num_roles,dtype=torch.long,device=triggerrole_indices.device)  # (batch_size, num_triggers * num_roles)

            # Calculate gold_arg_indices[triggerrole_indices[i]] = arg_span_indices[i] + 1 (index into `arg_spans` to recover actual span; +1 is to account for dummy "no argument")
            # gold_arg_indices is composed of `num_trigger_spans` number of blocks each of size `num_roles`; we place the arg span index (+1) at the positions specified by `triggerrole_indices`
            # `original_gold_arg_indices` basically says "in the gold data: for this trigger+role, we observed this argument span"
            # Shape: (batch_size, num_trigger_spans * num_roles) in range [num_args]
            original_gold_arg_indices = gold_arg_indices.scatter_(dim=1, index=triggerrole_indices, src=(pruned_target_arg_indices+1))

            # Shape: (batch_size, num_triggers, num_args + 1)
            # We have a map "the 8-th `best` span is index 200 in `arg_spans`", and now we need
            # the inverse of that ("index 200 in `arg_spans` is the 8-th `best` span"), offset by 1 for the dummy.
            # `inverse_sort_arg_indices` is the inverse map we desire, but does not incorporate the dummy.
            # This computes the offset-adjusted map...
            adjusted_inverse_sort_arg_indices = torch.cat([torch.zeros([1, num_trigger_spans, 1],
                                                                       dtype=torch.long,
                                                                       device=inverse_sort_arg_indices.device),
                                                           (1 + inverse_sort_arg_indices)], dim=2)

            # Shape: (num_trigger_spans * num_roles) in range [args_per_trigger]
            # ... and this pulls out the appropriate parts of the map
            remapped_gold_arg_indices = torch.gather(
                adjusted_inverse_sort_arg_indices,
                2,
                original_gold_arg_indices.view(-1, num_trigger_spans, num_roles)).view(-1) # assume batch size 1 in this reshape

            # Mask gold arguments that we had previously pruned to the number of args and the dummy
            remapped_gold_arg_indices *= remapped_gold_arg_indices.lt(max_args_per_trigger + 1).long()  # (num_trigger_spans * num_roles)
            unnormalized_loss = self._loss(input=link_scores, target=remapped_gold_arg_indices)  # torch.nn.CrossEntropyLoss expects *scores*
            output_dict["loss"] = unnormalized_loss

        if not for_training:
            predictions = self.decode(output_dict)
            try:
                # We have gold answers (dev, test)
                gold = self.gold(output_dict)
            except:
                # Making predictions on unlabeled data
                gold = None

            sentences = output_dict["metadata"][0]["sentences"]
            if gold:
                gold_data_path = self._test_gold_data_path if for_prediction else self._dev_gold_data_path
                # We have gold answers to compare against.
                # Only update scorer when not in train to save time.
                self._scorer(predictions, gold, sentences, gold_data_path)

                # beam_analysis = al_util.analyze_beam(output_dict)

            self.writeout(output_dict["task"], output_dict["metadata"][0], predictions, gold, sentences, for_prediction, outputdir, arg_spans[0, candidate_arg_indices].cpu().numpy().tolist())

        # Move self away from cpu
        if not for_training:
            self._revert_device_move(original_device, cpu_eval)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False, skip=False) -> Dict[str, float]:
        if skip:
            return {}
        return self._scorer.get_metric(reset)

    def gold(self, output_dict: Dict[str, torch.Tensor]):
        metadata = output_dict["metadata"][0]
        sentences, sentence_start_offsets, gold_triggers, gold_arguments, gold_roles = (metadata['sentences'],
                                                                                        metadata['sentence_start_offsets'],
                                                                                        metadata['triggers'],
                                                                                        metadata['arguments'],
                                                                                        metadata['roles'])
        try:
            gold_strings = metadata['gold_strings']
        except:
            gold_strings = ["@nogoldstring@"]*len(gold_arguments)

        sentence_lengths = [sum(sentence_mask) for sentence_mask in metadata['text_lens']]
        text = [word for sentence in sentences for word in sentence]
        confidences = [1]*len(gold_arguments)

        gold = al_util._combine_args(output_dict["task"], gold_triggers, gold_arguments, gold_roles, confidences, sentence_lengths, sentence_start_offsets, strings=gold_strings)  # [num_sentences]

        return gold

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        if self._for_prediction:
            # Make more predictions than necessary, and filter in post-processing
            decoder = "greedy"
        else:
            decoder = output_dict["decoder"]

        metadata = output_dict["metadata"][0]
        sentences, sentence_start_offsets = (metadata['sentences'],
                                             metadata['sentence_start_offsets'])
        sentence_lengths = [sum(sentence_mask) for sentence_mask in metadata['text_lens']]
        text = [word for sentence in sentences for word in sentence]
        if decoder == "argmax":
            # ASSUMPTION: selection of argument span for a given trigger+role is fully independent of other trigger+roles
            # ASSUMPTION: only one argument may fill a trigger+role slot
            decoded_predictions = al_util.argmax_decode(output_dict, text)
        elif decoder == "greedy":
            # ASSUMPTION: many arguments may fill a trigger+role slot
            decoded_predictions = al_util.greedy_decode(output_dict, text)
        else:
            raise ValueError(f"unrecognized decoder: {decoder}")

        if decoded_predictions:
            (predicted_triggers, predicted_args, predicted_roles, confs, predicted_strings) = zip(*decoded_predictions)
        else:
            (predicted_triggers, predicted_args, predicted_roles, confs, predicted_strings) = ([], [], [], [], [])

        predictions = al_util._combine_args(output_dict["task"], predicted_triggers, predicted_args, predicted_roles, confs, sentence_lengths, sentence_start_offsets, predicted_strings)  # [num_sentences]

        return predictions

    def _create_output_filename(self, doc_id, outputdir):
        return os.path.join(outputdir, f"results.{doc_id}")

    def writeout(self,
                 task: str,
                 metadata,
                 predictions,
                 gold,
                 sentences,
                 for_prediction,
                 outputdir,
                 candidate_args=None):
        # Report predictions
        doc_id = metadata["doc_id"]
        logger.info(f"{str(predictions)} \tPREDICTIONS ({doc_id})")
        logger.info(f"{str(gold)} \tGOLD ({doc_id})")

        if for_prediction:
            if task in ["rams"]:
                if not outputdir:
                    # Output to directory where input data is
                    outputdir = os.path.dirname(metadata["data_path"])
                output_file = os.path.join(outputdir, "predictions.jsonlines")
                if not os.path.isfile(output_file):
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)

                obj = dict()
                obj["doc_key"] = doc_id
                formatted_predictions = []
                for p_s in predictions["structures"]:
                    for trigger, args in p_s.items():
                        formatted_predictions.append([list(trigger), *[list(arg) for arg in args]])
                obj["predictions"] = formatted_predictions

                # JSON Lines format
                with open(output_file, "a") as f:
                    f.write(json.dumps(obj) + '\n')

            else:
                if not outputdir:
                    # Output to directory where input data is
                    outputdir = os.path.dirname(metadata["data_path"])
                output_file = self._create_output_filename(os.path.basename(metadata["data_path"]), outputdir)

                if not os.path.isfile(output_file):
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    shutil.copyfile(metadata["data_path"], output_file)

                # ASSUMPTION: JSON Lines format (each example on a separate line)
                logger.info(f"Writing model predictions to {output_file}...")
                objs = []
                replace_idx = None
                data = None
                with open(output_file, "r") as f:
                    for i,line in enumerate(f):
                        obj = json.loads(line)
                        objs.append(obj)
                        if obj["doc_key"] == doc_id:
                            data = obj
                            replace_idx = i

                if not data:
                    raise ValueError(f"Could not find an item with doc_id={doc_id}")

                formatted_predictions = []
                for p_s in predictions["structures"]:
                    for trigger, args in p_s.items():
                        formatted_predictions.append([list(trigger), *[list(arg) for arg in args]])

                data.update({f"{metadata['annotation_kind']}_predictions:{doc_id}": formatted_predictions})
                objs[replace_idx] = data

                with open(output_file, "w") as f:
                    for obj in objs:
                        f.write(json.dumps(obj) + '\n')


    def _load_contextualized_embs(self, tokens, genre, metadata, for_training):
        # ASSUMPTION: data for this batch comes from a single document
        full_lm_embs = self.single_lm_loader(metadata[0]["doc_id"], "train" if for_training else "test")  # (num_sent, max_sent_len, emb_size, num_layers)
        text_lens_mask = metadata[0]["text_lens"]  # (num_sent, max_sent_len)
        text_lens_mask_idx = text_lens_mask.reshape(-1).nonzero()  # [(document_length)]

        # Shape: (batch_size, num_sent, max_sent_len, emb_size)
        lm_embs = self._contextualized_embeddings_scalars(torch.as_tensor(full_lm_embs.transpose([3,0,1,2]),
                                                                          dtype=torch.float32, device=genre.device)).unsqueeze(0)
        num_sents = lm_embs.shape[1]  # int
        max_sent_len = lm_embs.shape[2]  # int
        document_length = text_lens_mask.sum()  # int
        # Shape: (batch_size, num_sent, max_sent_len, embedding_size)
        padded_text_embeddings = self._lexical_dropout(self._text_field_embedder(tokens)).reshape(1, num_sents, max_sent_len, -1)

        # Shape: (num_sent, max_sent_len, encoding_dim)
        padded_contextualized_embeddings = self._context_layer(torch.cat([padded_text_embeddings, lm_embs], dim=3).squeeze(0),
                                                               torch.tensor(text_lens_mask, device=padded_text_embeddings.device)) # pylint: disable=not-callable

        # Shape: (batch_size, document_length, encoding_dim)
        contextualized_embeddings = padded_contextualized_embeddings.reshape(num_sents * max_sent_len, -1)[text_lens_mask_idx].unsqueeze(0)

        # Shape: (batch_size, document_length, encoding_dim)
        text_embeddings = padded_text_embeddings.reshape(1, num_sents * max_sent_len, -1)[:, text_lens_mask_idx].squeeze(0)

        # Shape: (batch_size, feature_size)
        genre_emb = self._genre_embedding(genre)
        return contextualized_embeddings, text_embeddings, genre_emb, document_length


    def _compute_span_pair_embeddings(self,
                                      triggerrole_embeddings: torch.FloatTensor,
                                      arg_span_embeddings: torch.FloatTensor,
                                      token_distance_embeddings: torch.FloatTensor,
                                      genre_emb: torch.FloatTensor):
        """
        Computes an embedding representation of pairs of spans for the pairwise scoring function
        to consider. This includes both the original span representations, the element-wise
        similarity of the span representations, and an embedding representation of the distance
        between the two spans.

        Parameters
        ----------
        triggerrole_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the top spans. Has shape
            (batch_size, num_spans_to_keep, embedding_size).
        arg_span_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the partner spans we are considering
            for each top span. Has shape
            # (batch_size, num_spans_to_keep, max_partners, embedding_size).
            (batch_size, num_spans_to_keep, embedding_size).
        token_distance_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the distance between each top span and its partner spans
            in terms of spans we are considering. Has shape (1, num_triggers, max_args_per_trigger, embedding_size).

        Returns
        -------
        span_pair_embeddings : ``torch.FloatTensor``
            Embedding representation of the pair of spans to consider. Has shape
            (batch_size, num_spans_to_keep, max_partners, embedding_size)
        """
        args_per_trigger = arg_span_embeddings.size(-2)  # int
        num_triggers = triggerrole_embeddings.size(1)  # int
        num_roles = triggerrole_embeddings.size(2)  # int

        # Shape: (batch_size, num_triggers, num_roles, args_per_trigger, embedding_size)
        triggerrole_embeddings = triggerrole_embeddings.unsqueeze(3).repeat(1,1,1,args_per_trigger,1)
        arg_span_embeddings = arg_span_embeddings.unsqueeze(2).repeat(1,1,num_roles,1,1)

        span_emb_list = [triggerrole_embeddings,
                         arg_span_embeddings,
                         triggerrole_embeddings * arg_span_embeddings]

        if self._span_pair_feats['use_token_distances']:
            # Shape: (batch_size, num_triggers, num_roles, args_per_trigger, embedding_size)
            span_emb_list.append(token_distance_embeddings.unsqueeze(2).repeat(1,1,num_roles,1,1))
        if self._span_pair_feats['use_genre']:
            # Shape: (batch_size, num_triggers, num_roles, args_per_trigger, embedding_size)
            span_emb_list.append(genre_emb.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,num_triggers,num_roles,args_per_trigger,1))

        span_pair_embeddings = torch.cat(span_emb_list, -1) # axis

        return span_pair_embeddings


    def _compute_link_scores(self,
                             num_triggers: int,
                             num_args: int,
                             num_roles: int,
                             pairwise_embeddings: torch.FloatTensor,
                             triggerrole_scores: torch.FloatTensor,
                             argrole_scores: torch.FloatTensor,
                             trigger_mention_scores: torch.FloatTensor,
                             arg_mention_scores: torch.FloatTensor,
                             trigger_arg_scores: torch.FloatTensor,
                             log_mask: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        """
        Computes scores for every pair of spans. Additionally, a dummy label is included,
        representing the decision that the span is not linked with anything. For the dummy
        label, the score is always zero. For the true linked spans, the score consists of
        the pairwise mention score and the unary mention scores for the span and its
        linked partner. The factoring allows the model to blame many of the absent links on bad
        spans, enabling the pruning strategy used in the forward pass.

        Parameters
        ----------
        pairwise_embeddings: ``torch.FloatTensor``, required.
            Embedding representations of pairs of spans. Has shape
            (batch_size, num_triggers, num_roles, num_args, encoding_dim)
        triggerrole_scores: ``torch.FloatTensor``, required.
            Mention scores for every span. Has shape
            (batch_size, num_triggers, num_roles).
        trigger_mention_scores: ``torch.FloatTensor``, required.
            Mention scores for every partner. Has shape
            (batch_size, num_triggers).
        arg_mention_scores: ``torch.FloatTensor``, required.
            Mention scores for every argument span. Has shape
            (batch_size, num_args).
        log_mask: ``torch.FloatTensor``, required.
            The log of the mask for valid partners.

        Returns
        -------
        link_scores: ``torch.FloatTensor``
            A tensor of shape (batch_size, num_triggers, num_roles, num_args + 1),
            representing the unnormalised score for each (trigger, role, argument) triple
            we considered.

        """
        # Shape: (batch_size, num_triggers, num_roles, num_args(per trigger))
        link_scores = torch.zeros(1, num_triggers, num_roles, num_args,
                                  device=trigger_mention_scores.device)

        # Heavy use of broadcasting
        if self._link_scores_params['use_trigger_scores']:
            # Shape: (batch_size, num_triggers, 1, 1)
            link_scores += trigger_mention_scores.unsqueeze(2).unsqueeze(3)
        if self._link_scores_params['use_arg_scores']:
            # Shape: (batch_size, num_triggers, 1, args_per_trigger)
            link_scores += arg_mention_scores.unsqueeze(2).squeeze(-1)
        if self._link_scores_params['use_triggerrole_scores']:
            # Shape: (batch_size, num_triggers, num_roles, 1)
            link_scores += triggerrole_scores.unsqueeze(3)
        if self._link_scores_params['use_argrole_scores']:
            # Shape: (batch_size, num_triggers, num_roles, num_args)
            link_scores += argrole_scores
        if self._link_scores_params['use_coarse_scores']:
            # Shape: (batch_size, num_triggers, 1, args_per_trigger)
            link_scores += trigger_arg_scores.unsqueeze(-2) # Add coarse scores
        if self._link_scores_params['use_partner_scores']:
            # Shape: (batch_size, num_triggers, num_roles, args_per_trigger)
            link_scores += self._link_scorer(pairwise_embeddings).squeeze(-1)
        if self._link_scores_params['apply_log_mask'] and log_mask is not None:
            link_scores += log_mask

        # Dummy is fixed to 0 and represents the option to choose when there is no good link (i.e., all scores are negative)
        # Shape: (batch_size, num_triggers, num_roles, 1)
        shape = [link_scores.size(0), num_triggers, num_roles, 1]
        dummy_scores = link_scores.new_zeros(*shape)

        # Shape: (batch_size, num_triggers, num_roles, num_args + 1)
        link_scores_with_dummy = torch.cat([dummy_scores, link_scores], -1)
        return link_scores_with_dummy

    def _find_top_args_for_trigger(self, trigger_arg_scores, max_args_per_trigger):
        """
        Similar to top_k but also returns an inverse. We need to full idx_of_sorted to
        correctly remap target labels. Takes in scores computed elsewhere.

        Sorted such that *higher* scores are better.

        Possible ways of scoring:
          * absolute distance
          * restricted absolute distance (must be same-sentence)
          * coarse pruner.
        See trigger_arg_scores for more information
        trigger_arg_scores has shape (batch_size, num_pres, num_args)
        """
        sorted_trigger_arg_scores, idx_of_sorted = torch.sort(trigger_arg_scores, dim=2, descending=True)  # (batch_size, num_triggers, num_args)

        # Do the pruning
        candidate_trigger_arg_scores = sorted_trigger_arg_scores[:, :, :max_args_per_trigger]  # (batch_size, num_triggers, max_args_per_trigger)
        candidate_arg_indices = idx_of_sorted[:, :, :max_args_per_trigger].contiguous().squeeze(0)  # (num_triggers, max_args_per_trigger)

        # We have a map (`sort_indices`) that says "the 8-th `closest` span is index 200 in `arg_spans`" (its 8th element == 200).
        # Now we need the inverse of that, in other words, "index 200 in `arg_spans` is the 8-th `closest` span" (its 200th element == 8).
        # `inverse_sort_arg_indices` is the inverse map we desire, and we can compute it using argsort() because indices are unique.
        inverse_sort_arg_indices = torch.argsort(idx_of_sorted, dim=2)

        return candidate_trigger_arg_scores, candidate_arg_indices, inverse_sort_arg_indices


    def _get_coarse_scores(self, trigger_span_embeddings, arg_span_embeddings):
        """
        Does (trigger * W) * args as a simple bilinear scoring function
        """
        # (batch_size, num_triggers, trigger_emb_dim) * (trigger_emb_dim, arg_emb_dim) * (batch_size, arg_emb_dim, num_args)
        # Results in (batch_size, num_triggers, num_args)
        dropout = torch.nn.Dropout(p=self._default_dropout)
        return (trigger_span_embeddings
                .matmul(dropout(self._coarse_scorer))
                .matmul(arg_span_embeddings.transpose(-1, -2)))

    def _device_move(self, batch, cpu_eval):
        original_device = batch['genre'].device
        if (cpu_eval and batch['metadata'][0]['text_lens'].sum() > self._max_inference_doc_size):
            self.to('cpu')
            batch = move_to_device(batch, 'cpu')
        return (original_device, batch)

    def _revert_device_move(self, device, cpu_eval):
        if cpu_eval:
            self.to(device)

    def _maybe_empty_cache(self, batch, device, train=True):
        if self._empty_cache:
            torch.cuda.empty_cache()

    @classmethod
    def from_params(cls, vocab, params):
        logger.info('Building ArgLinking Model...')
        task = params['task_name']
        if vocab._oov_token not in vocab._token_to_index["labels"]:
            vocab.add_token_to_namespace(vocab._oov_token, "labels")

        logger.info(vocab._token_to_index["labels"])

        _, trigger_dim, arg_dim, num_roles = al_util.autopopulate_params(vocab, params) # util function
        logger.info("Params autopopulated: trigger_dim: {}, arg_dim: {}, trigger_role_dim: {}, arg_role_dim: {}, big_dim: {}, num_roles: {}".format(
            trigger_dim,
            arg_dim,
            params['trigger_role_feedforward']['input_dim'],
            params['arg_role_feedforward']['input_dim'],
            params['link_feedforward']['input_dim'],
            num_roles))

        token_embedders = {
            "tokens": Embedding.from_params(vocab, params['token_embedders']['tokens']),
            "token_characters": TokenCharactersEncoder.from_params(vocab, params['token_embedders']['token_characters']),
        }
        text_field_embedder = BasicTextFieldEmbedder(token_embedders)
        context_layer_params = params['context_layer']

        if params['recurrent_dropout']:
            context_layer = PytorchSeq2SeqWrapper(
                module=StackedBidirectionalLstm(input_size=context_layer_params['input_size'],
                                                hidden_size=context_layer_params['hidden_size'],
                                                num_layers=context_layer_params['num_layers'],
                                                recurrent_dropout_probability=context_layer_params['dropout']),
                stateful=False
                )
        else:
            context_layer = PytorchSeq2SeqWrapper(
                module=torch.nn.LSTM(**params['context_layer'], batch_first=True),
                stateful=False
            )

        dev_gold_data_path = params['dev_gold_data_path']
        test_gold_data_path = params['test_gold_data_path']

        arch_params = params['arch_params']

        require_triggerrole_scores = (arch_params['link_scores']['use_triggerrole_scores'])
        require_argrole_scores = (arch_params['link_scores']['use_argrole_scores'])
        require_token_distances = (arch_params['trigger_arg_components']['token_distances'] or arch_params['link_scores']['use_partner_scores'] or arch_params['span_pair_feats']['use_token_distances'])

        require_coarse_scorer = (arch_params['trigger_arg_components']['coarse_score'])
        require_trigger_role_feedforward_scorer = require_triggerrole_scores
        require_arg_role_feedforward_scorer = require_argrole_scores
        require_link_scorer = (arch_params['link_scores']['use_partner_scores'])
        require_distance_scorer = require_token_distances

        required_components = {"require_triggerrole_scores": require_triggerrole_scores,
                               "require_argrole_scores": require_argrole_scores,
                               "require_token_distances": require_token_distances,
                               "require_coarse_scorer": require_coarse_scorer,
                               "require_trigger_role_feedforward_scorer": require_trigger_role_feedforward_scorer,
                               "require_arg_role_feedforward_scorer": require_arg_role_feedforward_scorer,
                               "require_link_scorer": require_link_scorer,
                               "require_distance_scorer": require_distance_scorer}


        trigger_mention_feedforward = FeedForward.from_params(params['trigger_mention_feedforward'])
        arg_mention_feedforward = FeedForward.from_params(params['arg_mention_feedforward'])
        if require_trigger_role_feedforward_scorer:
            trigger_role_feedforward = FeedForward.from_params(params['trigger_role_feedforward'])
        else:
            trigger_role_feedforward = None
        trigger_role_projection = FeedForward.from_params(params['trigger_role_projection'])
        if require_arg_role_feedforward_scorer:
            arg_role_feedforward = FeedForward.from_params(params['arg_role_feedforward'])
        else:
            arg_role_feedforward = None
        if require_link_scorer:
            link_feedforward = FeedForward.from_params(params['link_feedforward'])
        else:
            link_feedforward = None
        if require_distance_scorer:
            distance_feedforward = FeedForward.from_params(params['distance_feedforward'])

        max_args_per_trigger = params['max_args_per_trigger']
        num_genres = len(params['genres'])
        num_widths = params['num_widths']
        trigger_spans_per_word = params['trigger_spans_per_word']
        arg_spans_per_word = params['arg_spans_per_word']
        max_training_doc_size = params['max_training_doc_size']
        max_inference_doc_size = params.get('max_inference_doc_size',
                                            max_training_doc_size)
        empty_cache = params.get('empty_cache', True)
        default_dropout = params['default_dropout']
        role_embeddings = Parameter(torch.FloatTensor(num_roles,
                                                      params['role_dim']))

        if require_coarse_scorer:
            coarse_scorer = Parameter(torch.FloatTensor(trigger_dim,
                                                        arg_dim))
        else:
            coarse_scorer = None

        feature_size = params['feature_size']
        lexical_dropout = params['lexical_dropout']

        lm_file = h5py.File(params['lm_file'], 'r')
        test_lm_file = h5py.File(params['test_lm_file'], 'r')
        lm_layers = params['lm_layers']
        lm_size = params['lm_size']
        lm_loaders = {"train": lambda doc_key: load_lm_embeddings(doc_key, lm_file, lm_size, lm_layers),
                      "test": lambda doc_key: load_lm_embeddings(doc_key, test_lm_file, lm_size, lm_layers)}

        # initializers not passed in in config
        initializer = InitializerApplicator.from_params(list(map(tuple,
                                                            [
                                                                [".*linear_layers.*weight", {"type": "xavier_normal"}],
                                                                [".*_scorer.1._module.weight", {"type": "xavier_normal"}],
                                                                ["_distance_embedding.weight", {"type": "xavier_normal"}],
                                                                ["_span_width_embedding.weight", {"type": "xavier_normal"}],
                                                                ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
                                                                ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}],
                                                                ["_role_embeddings", {"type": "xavier_normal"}],
                                                                ["_coarse_scorer", {"type": "xavier_normal"}],
                                                            ])))

        model = ArgLinkingModel(task=task,
                                vocab=vocab,
                                dev_gold_data_path=dev_gold_data_path,
                                test_gold_data_path=test_gold_data_path,
                                required_components=required_components,
                                text_field_embedder=text_field_embedder,
                                context_layer=context_layer,
                                trigger_mention_feedforward=trigger_mention_feedforward,
                                arg_mention_feedforward=arg_mention_feedforward,
                                trigger_role_feedforward=trigger_role_feedforward,
                                trigger_role_projection=trigger_role_projection,
                                arg_role_feedforward=arg_role_feedforward,
                                link_feedforward=link_feedforward,
                                distance_feedforward=distance_feedforward,
                                role_embeddings=role_embeddings,
                                num_widths=num_widths,
                                max_args_per_trigger=max_args_per_trigger,
                                trigger_spans_per_word=trigger_spans_per_word,
                                arg_spans_per_word=arg_spans_per_word,
                                coarse_scorer=coarse_scorer,
                                max_training_doc_size=max_training_doc_size,
                                max_inference_doc_size=max_inference_doc_size,
                                num_genres=num_genres,
                                feature_size=feature_size,
                                lm_loaders=lm_loaders,
                                lm_layers=lm_layers,
                                arch_params=arch_params,
                                empty_cache=empty_cache,
                                default_dropout=default_dropout,
                                lexical_dropout=lexical_dropout,
                                initializer=initializer)
        logger.info(model)
        return model
