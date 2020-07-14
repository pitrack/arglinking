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
import functools
import json


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

CROSSES_SENTENCE_BOUNDARY = -999

@DatasetReader.register("gvdb")
class GVDBDatasetReader(DatasetReader):
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
        self._file_path = None
        self._genres = genres

    def _read_document(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        if self._file_path is None:
            self._file_path = file_path
        file_path = cached_path(file_path)
        logger.info("Reading GVDB instances from dataset files at: %s", file_path)

        with open(file_path, "r", encoding="utf-8") as text_file:
            reports = text_file.read().splitlines()

        unique_slots = set()
        reports_skipped = 0
        args_skipped = 0
        input_args_seen = 0
        for i, reportline in enumerate(reports):
            report = json.loads(reportline)

            tokens = []
            token_offset = 0  # word offset wrt the document
            sentence_offsets = []
            text_lens = []

            date = report["date"]
            sentences = report["full_text"]
            spans = report["spans"]

            # During training, we want to skip documents without arguments.
            # During test, we do not. This flag is hard to pass through to
            # the dataset reader, and so instead we directly  modify the code.
            # It's possible that it's safe to always append @@UNKNOWN@@, but
            # that will affect model behavior.
            training = False

            if spans == []:
                # No annotations for this report, so skip it
                reports_skipped += 1
                if training:
                    continue
                else:
                    spans.append([0,1,"@@UNKNOWN@@", "@@@", ["@@@"]])

            for sentence in sentences:
                sentence_offsets.append(token_offset)
                tokens.extend([word for word in sentence])
                token_offset += len(sentence)
                text_lens.append(len(sentence))

            all_links = []  # elements are (event_type, slot, value) tuples
            doc_start, doc_end = 0, len(tokens) - 1  # [0, num_tokens) -> [0, num_tokens - 1] to form an inclusive interval
            for link in spans:
                v_start, v_end, slot, gold, _ = link
                v_end -= 1  # right end is exclusive, so make it inclusive
                # the trigger is the entire document
                (t,s,v,g) = ((doc_start,doc_end), slot, (v_start,v_end), gold)
                all_links.append((t,s,v,g))
                unique_slots.add(slot)
                input_args_seen += 1

            text = " ".join(tokens)
            genre = None
            assert len(sentence_offsets) == len(text_lens)
            doc_id = report.get("doc_key", file_path + "_" + str(i))

            doc_link_info = dict()  # elements are {event_type: [(value, slot)]} dicts
            for (t,s,v,g) in all_links:
                if t not in doc_link_info:
                    doc_link_info[t] = {t: [(s, v, g)]}
                else:
                    doc_link_info[t][t].extend([(s, v, g)])

            doc_link_info = [*doc_link_info.values()]

            # Find if report has any within-sentence value spans (i.e., findable by our model)
            # If it doesn't, skip this report
            trigger_arg_pairs = [] if doc_link_info is not None else None
            args = [] if doc_link_info is not None else None
            arg_sent_ids = {} if doc_link_info is not None else None
            if doc_link_info is not None:
                trigger_arg_pairs = []
                for frame_data in doc_link_info:
                    for trigger_span, argument_data in frame_data.items():
                        for (role, argument_span, gold) in argument_data:
                            trigger_arg_pairs.append([trigger_span, argument_span, role, gold])
                            arg_sentence_id = self._get_sentence_id(argument_span, sentence_offsets+[float('inf')])
                            if arg_sentence_id == CROSSES_SENTENCE_BOUNDARY:
                                args_skipped += 1
                                continue

                            args.append(argument_span)

                            arg_sent_ids[argument_span] = arg_sentence_id
            if len(args) == 0:
                logger.info("No values for report #{}".format(i))
                reports_skipped += 1
                continue

            yield self.text_to_instance(sentences,
                                        sentence_offsets,
                                        doc_link_info,
                                        genre,
                                        doc_id)
        logger.info("Unique slots: {}".format(sorted(unique_slots)))
        logger.info("Skipped {} reports".format(reports_skipped))
        logger.info("Skipped {} arguments because they crossed a sentence boundary".format(args_skipped))
        logger.info("{} arguments were given in the input documents".format(input_args_seen))

    @overrides
    def _read(self, file_path: str):
        return self._read_document(file_path)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentences: List[List[str]],
                         sentence_start_offsets: List[int],
                         doc_link_info: Optional[List[Dict[Tuple[int, int], List[Tuple[str, Tuple[int, int], str]]]]] = None,  # [{(event_type_start, event_type_end): [(slot, (value_start, value_end))]}]
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
        #                    labels      (trigger,arg) span pairs
        trigger_arg_pairs = [] if doc_link_info is not None else None
        triggers = [] if doc_link_info is not None else None
        args = [] if doc_link_info is not None else None
        roles = [] if doc_link_info is not None else None
        gold_strings = [] if doc_link_info is not None else None
        unique_roles = set() if doc_link_info is not None else None
        trigger_sent_ids = {} if doc_link_info is not None else None
        arg_sent_ids = {} if doc_link_info is not None else None
        skipped_args = 0
        if doc_link_info is not None:
            trigger_arg_pairs = []
            seen_trigger_spans = set()
            seen_arg_spans = set()
            for frame_data in doc_link_info:
                for trigger_span, argument_data in frame_data.items():
                    trigger_sentence_id = self._get_sentence_id(trigger_span, sentence_offsets)
                    # event_type is allowed to cross sentence boundaries, since the event_type is the entire document
                    trigger_sent_ids[trigger_span] = trigger_sentence_id
                    for (role, argument_span, gold_string) in argument_data:
                        trigger_arg_pairs.append([trigger_span, argument_span, role, gold_string])
                        arg_sentence_id = self._get_sentence_id(argument_span, sentence_offsets)

                        if arg_sentence_id == CROSSES_SENTENCE_BOUNDARY:
                            skipped_args += 1
                            continue

                        # If not using gold, need to make sure the labels
                        # we care about exist in the enumerated spans
                        span_size_condition = True
                        if (not self._use_gold_triggers):
                            span_size_condition &= (trigger_span[1] - trigger_span[0] < self._max_trigger_span_width)
                        if (not self._use_gold_arguments):
                            span_size_condition &= (argument_span[1] - argument_span[0] < self._max_arg_span_width)
                        if span_size_condition:
                            roles.append(LabelField(role))
                            unique_roles.add(role)
                            triggers.append(trigger_span)
                            args.append(argument_span)

                            arg_sent_ids[argument_span] = arg_sentence_id

        metadata: Dict[str, Any] = dict()
        metadata["annotation_kind"] = "" # to play nicely with RAMS
        metadata["sentences"] = sentences
        metadata["sentence_start_offsets"] = sentence_start_offsets
        metadata["text_lens"] = np.array(text_lens_mask)
        metadata["doc_id"] = document_id
        metadata["has_gold_targets"] = doc_link_info is not None
        metadata["data_path"] = self._file_path

        # Create all triggers and all args
        # Since the event_type is the entire document (and therefore crosses boundaries), we replace CROSSES_SENTENCE_BOUNDARY with sentence id of 0 to ensure we don't get out-of-bounds issues
        trigger_sent_ids = {k: (lambda v: v if v != CROSSES_SENTENCE_BOUNDARY else 0)(v) for k, v in trigger_sent_ids.items()}
        trigger_spans_field, trigger_idx_field, trigger_sentence_id_field = self._get_unique_spans(triggers, text_field, trigger_sent_ids)
        if self._use_gold_arguments:
            arg_spans_field, arg_idx_field, arg_sentence_id_field = self._get_unique_spans(args, text_field, arg_sent_ids)
        else:
            arg_spans_field, arg_idx_field, arg_sentence_id_field = self._enumerate_spans(sentences, self._max_arg_span_width, text_field, args)

        (metadata['triggers'], metadata['arguments'], metadata['roles'], metadata['gold_strings']) = zip(*trigger_arg_pairs)
        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {
            "text": text_field,
            "genre": genre_field,
            "metadata": metadata_field,
            "all_triggers": trigger_spans_field,
            "all_args": arg_spans_field,
            "all_trigger_sentence_ids": trigger_sentence_id_field,
            "all_arg_sentence_ids": arg_sentence_id_field,
            "target_roles": ListField(roles),
            "target_trigger_idx": trigger_idx_field,
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
        # span_indices are effectively the labels
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

        return CROSSES_SENTENCE_BOUNDARY
