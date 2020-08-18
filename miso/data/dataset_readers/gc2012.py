import logging
from typing import Dict, List, Tuple, Iterable, Union, Any, Optional

from overrides import overrides

from miso.utils.file import cached_path
from miso.data.dataset_readers.dataset_reader import DatasetReader
from miso.data.fields import Field, TextField, SequenceLabelField, MetadataField, ListField, SpanField, LabelField, IndexField
from miso.data.instance import Instance
from miso.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from miso.data.tokenizers import Token
from miso.data.dataset_readers.dataset_utils import enumerate_spans

import numpy as np
import json
from collections import defaultdict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("gc2012srl")
class GerberChai2012SRLDatasetReader(DatasetReader):
    def __init__(self,
                 max_trigger_span_width: int,
                 max_arg_span_width: int,
                 use_gold_triggers: bool,
                 use_gold_arguments: bool,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 genres: List[str] = []) -> None:
        super().__init__(lazy)
        self._max_trigger_span_width = max_trigger_span_width
        self._max_arg_span_width = max_arg_span_width
        self._use_gold_triggers = use_gold_triggers
        self._use_gold_arguments = use_gold_arguments
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._genres = {g: i for i, g in enumerate(genres)}
        self._file_path = None

    def _read_document(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        if self._file_path is None:
            self._file_path = file_path
        file_path = cached_path(file_path)
        logger.info("Reading GC2012 instances from dataset file at: %s", file_path)

        # See `http://lair.cse.msu.edu/projects/implicit_annotations.html` for details.

        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))

        for example in examples:
            pred_arg_info = defaultdict(list)
            trigger_span = tuple(example["trigger"]["span"])
            for argn in example["arguments"]:
                arguments = example["arguments"][argn]
                # assert len(arguments) == 1  # expecting just the closest argument to trigger rather than a whole argument cluster
                for a in arguments:
                    argument_span = a["span"]

                    pred_arg_info[trigger_span].append((argn, tuple(argument_span)))

            if pred_arg_info == dict():
                # no annotations for this example
                continue

            sentence_start_offsets = []
            sentence_ids = []
            total_tokens = 0
            for s_id,sentence in enumerate(example["sentences"]):
                sentence_start_offsets.append(total_tokens)
                sentence_ids.append(s_id)

                total_tokens += len(sentence)

            doc_id = example["doc_key"]
            genre = 'nw'
            yield self.text_to_instance(example["sentences"],
                                        sentence_start_offsets,
                                        pred_arg_info,
                                        genre,
                                        doc_id)


    @overrides
    def _read(self, file_path: str):
        return self._read_document(file_path)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentences: List[List[str]],
                         sentence_start_offsets: List[int],
                         pred_arg_info: Optional[Dict[Tuple[int, int], List[Tuple[str, Tuple[int, int]]]]] = None,
                         genre: Optional[str] = None,
                         document_id: Optional[str] = None) -> Instance:
        # pylint: disable=arguments-differ
        max_sent_len = max(len(s) for s in sentences)
        padded_sentences = [[self._normalize_word(sentence[i]) if i < len(sentence) else "UNK"
                             for i in range(max_sent_len)]
                            for sentence in sentences]
        flattened_sentences = [word for sentence in padded_sentences
                               for word in sentence]

        text_lens = [len(s) for s in sentences]
        text_lens_idx = [list(range(max_sent_len)) for _ in sentences]
        text_lens_mask = [[int(idx < sent_len) for idx in idxs]
                          for sent_len, idxs in zip(text_lens, text_lens_idx)]
        sentence_offsets = [0]
        for tl in text_lens:
            sentence_offsets.append(sentence_offsets[-1] + tl)
        sentence_offsets = sentence_offsets[:-1]
        sentence_offsets.append(float('inf'))  # sentinel/padding for finding sentence id from span

        text_field = TextField([Token(word) for word in flattened_sentences], self._token_indexers)

        if genre:
            genre_field = LabelField(self._genres[genre], skip_indexing=True)
        else:
            genre_field = LabelField(0, skip_indexing=True)

        # pack up data as: List[str], List[List[SpanField]]
        #                       |               |
        #                    labels      (pred,arg) span pairs
        pred_arg_pairs = [] if pred_arg_info is not None else None
        preds = [] if pred_arg_info is not None else None
        args = [] if pred_arg_info is not None else None
        roles = [] if pred_arg_info is not None else None
        pred_sent_ids = {} if pred_arg_info is not None else None
        arg_sent_ids = {} if pred_arg_info is not None else None
        if pred_arg_info is not None:
            pred_arg_pairs = []
            seen_pred_spans = set()
            seen_arg_spans = set()
            for pred_span, argument_data in pred_arg_info.items():
                pred_sentence_id = self._get_sentence_id(pred_span, sentence_offsets)
                pred_sent_ids[pred_span] = pred_sentence_id
                for (role, argument_span) in argument_data:
                    pred_arg_pairs.append([pred_span, argument_span, role])
                    arg_sentence_id = self._get_sentence_id(argument_span, sentence_offsets)

                    # If not using gold, need to make sure the labels
                    # we care about exist in the enumerated spans
                    span_size_condition = True
                    if (not self._use_gold_triggers):
                        span_size_condition &= (pred_span[1] - pred_span[0] < self._max_trigger_span_width)
                    if (not self._use_gold_arguments):
                        span_size_condition &= (argument_span[1] - argument_span[0] < self._max_arg_span_width)
                    if span_size_condition:
                        roles.append(LabelField(role))
                        preds.append(pred_span)
                        args.append(argument_span)

                        arg_sent_ids[argument_span] = arg_sentence_id

        metadata: Dict[str, Any] = dict()
        metadata["sentences"] = sentences
        metadata["sentence_start_offsets"] = sentence_start_offsets
        metadata["text_lens"] = np.array(text_lens_mask)
        metadata["doc_id"] = document_id
        metadata["has_gold_targets"] = pred_arg_info is not None
        metadata["data_path"] = self._file_path
        metadata["annotation_kind"] = "SRL"

        # Create all preds and all args
        if self._use_gold_triggers:
            pred_spans_field, pred_idx_field, pred_sentence_id_field = self._get_unique_spans(preds, text_field, pred_sent_ids)
        else:
            pred_spans_field, pred_idx_field, pred_sentence_id_field = self._enumerate_spans(sentences, self._max_trigger_span_width, text_field, preds)
        if self._use_gold_arguments:
            arg_spans_field, arg_idx_field, arg_sentence_id_field = self._get_unique_spans(args, text_field, arg_sent_ids)
        else:
            arg_spans_field, arg_idx_field, arg_sentence_id_field = self._enumerate_spans(sentences, self._max_arg_span_width, text_field, args)

        (metadata['triggers'], metadata['arguments'], metadata['roles']) = zip(*pred_arg_pairs)
        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {
            "text": text_field,
            "genre": genre_field,
            "metadata": metadata_field,
            "all_triggers": pred_spans_field,
            "all_args": arg_spans_field,
            "all_trigger_sentence_ids": pred_sentence_id_field,
            "all_arg_sentence_ids": arg_sentence_id_field,
            "target_roles": ListField(roles),
            "target_trigger_idx": pred_idx_field,
            "target_arg_idx": arg_idx_field,
            }

        return Instance(fields)

    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word

    def _get_unique_spans(self, spans, text, sentence_ids):
        # span_indices are effectives the labels
        spans, span_indices = np.unique(np.array(spans), return_inverse=True, axis=0)
        spans_list_field = ListField([SpanField(int(s[0]), int(s[1]), text) for s in spans])
        labels_index_field = ListField([IndexField(int(idx), spans_list_field) for idx in span_indices])

        sent_ids = [sentence_ids[(s[0], s[1])] for s in spans]
        sentence_id_field = ListField([LabelField(si, label_namespace="sentence_id_tags", skip_indexing=True) for si in sent_ids])

        return spans_list_field, labels_index_field, sentence_id_field

    def _enumerate_spans(self, sentences, max_span_width, text, labels_list):
        """
        Enumerates all spans
        """
        spans: List[Field] = []
        sentence_offset = 0
        python_spans = []
        sentence_ids = []
        for sentence_id,sentence in enumerate(sentences):
            for start, end in enumerate_spans(sentence,
                                              offset=sentence_offset,
                                              max_span_width=max_span_width):
                spans.append(SpanField(start, end, text))
                python_spans.append((start, end))
                sentence_ids.append(LabelField(sentence_id, label_namespace="sentence_id_tags", skip_indexing=True))
            sentence_offset += len(sentence)
        python_span_dict = {span:i for i, span in enumerate(python_spans)}
        labels_idx = [python_span_dict[span] for span in labels_list]
        spans_list_field = ListField(spans)
        labels_index_field = ListField([IndexField(idx, spans_list_field) for idx in labels_idx])

        sentence_id_field = ListField(sentence_ids)
        return spans_list_field, labels_index_field, sentence_id_field

    def _get_sentence_id(self, span, sentence_offsets):
        start = span[0]
        end = span[1]

        for i in range(len(sentence_offsets)-1):
            if (start >= sentence_offsets[i]) and (end < sentence_offsets[i+1]):
                # right-exclusive: sentence_offsets[i+1] marks the index of the first token of sentence i+1
                return i
