from typing import (Dict, List, Tuple, Union, Iterator, Optional)
import logging

from overrides import overrides

from miso.utils.file import cached_path
from miso.data.instance import Instance
from miso.data.tokenizers import Token, WordTokenizer
from miso.data.tokenizers.word_splitter import JustSpacesWordSplitter
from miso.data.dataset_readers.dataset_reader import DatasetReader
from miso.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from miso.data.fields import (Field,
                              TextField,
                              MetadataField,
                              SequenceLabelField,
                              SpanField,
                              LabelField,
                              ListField,
                              IndexField)
from miso.data.dataset_readers.dataset_utils import enumerate_spans
from miso.data.vocabulary import DEFAULT_OOV_TOKEN

import numpy as np
import json
from collections import defaultdict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

ANNOTATION_MODES = ["events"]
CROSSES_SENTENCE_BOUNDARY = -999
ROLE_MAP = lambda x: x[11:]  # role has an 11-character (fixed length) prefix (e.g., `evt008arg04place` -> `place`)

REMOVE = []

def deep_tuple(x):
    """fully copies trees of lists to a tree of tuples.
       deep_tuple( [1,2,[3,4]] ) returns (1,2,(3,4))"""
    if type(x)!=type( [] ):
        return x
    return tuple(map(deep_tuple,x))

@DatasetReader.register("rams_arglinking")
class RAMSDatasetReader(DatasetReader):
    def __init__(self,
                 max_trigger_span_width: int,
                 max_arg_span_width: int,
                 use_gold_triggers: bool,
                 use_gold_arguments: bool,
                 annotation_mode: str,
                 language: str,
                 genres: List[str],
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._max_trigger_span_width = max_trigger_span_width
        self._max_arg_span_width = max_arg_span_width
        self._use_gold_triggers = use_gold_triggers
        self._use_gold_arguments = use_gold_arguments
        self._annotation_mode = annotation_mode
        self._language = language
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._genres = {g: i for i, g in enumerate(genres)}
        self._file_path = None
        self._tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())  # no-op tokenizer, since data is pre-tokenized

        if self._annotation_mode not in ANNOTATION_MODES:
            raise ValueError(f"unrecognized annotation mode: {self._annotation_mode}")

    def _read_from_json(self, file_paths: List[str]) -> Iterator[Instance]:
        for file_path in file_paths:
            # if `file_path` is a URL, redirect to the cache
            self._file_path = file_path
            file_path = cached_path(file_path)
            skip_if_no_gold_links = True  #### !!!!! When not given both gold triggers and gold arguments: Set to True during training, False during prediction

            with open(file_path, "r") as f:
                input_ = json.load(f)

            # check if document is in the language we expect
            doc_lang = input_["language_id"]
            if doc_lang != self._language:
                logger.info(f"Skipping file because it isn't in the target language: {file_path}")
                continue

            sentences = input_["sentences"]

            sentence_start_offsets = []
            total_tokens = 0
            for sentence in sentences:
                sentence_start_offsets.append(total_tokens)
                total_tokens += len(sentence)

            if self._annotation_mode == "events":
                triggers = input_["evt_triggers"]
                #triggers = [[0, total_tokens-1, t[2]] for t in triggers] # baseline: full-context trigger

                # Entity mentions and event mentions can be arguments
                arguments = input_["ent_spans"] + input_["evt_triggers"]

                has_gold_links = "gold_evt_links" in input_

                if has_gold_links:
                    links = input_["gold_evt_links"]
                    #links = [[[0, total_tokens-1], link[1], link[2]] for link in links] # baseline: full-context trigger
                else:
                    links = []

                if skip_if_no_gold_links and links == []:
                    # No links in this training document, so skip it
                    logger.info(f"Skipping file because it has no links: {file_path}")
                    continue

                recoverable_triggers = [[t[0], t[1]] for t in triggers if (t[1] - t[0])+1 <= self._max_trigger_span_width]
                recoverable_arguments = [[a[0], a[1]] for a in arguments if (a[1] - a[0])+1 <= self._max_arg_span_width]
                recoverable_links = [l for l in links]
                recoverable_links = [rl for rl in recoverable_links if (self._use_gold_triggers or rl[0] in recoverable_triggers) and (self._use_gold_arguments or rl[1] in recoverable_arguments)]
                if skip_if_no_gold_links and len(recoverable_links) == 0:
                    logger.info(f"Skipping file because it has no RECOVERABLE links (max trigger width={self._max_trigger_span_width}, max arg width={self._max_arg_span_width}): {file_path}")
                    continue

            else:
                raise ValueError(f"unrecognized annotation mode: {self._annotation_mode}")

            # Extract just the spans and not any auxiliary information (e.g., types)
            trigger_spans = tuple((t[0], t[1]) for t in triggers)
            argument_spans = tuple((a[0], a[1]) for a in arguments)

            # Deduplicate spans so that elements of `trigger_spans` and `argument_spans` are unique
            trigger_spans = list(set(trigger_spans))
            argument_spans = list(set(argument_spans))
            links = list(deep_tuple(links))

            assert len(set(tuple(trigger_spans))) == len(trigger_spans)
            assert len(set(tuple(argument_spans))) == len(argument_spans)

            # Annotations might not be in sorted order,
            # so sort by start index, then by end index, then by type
            trigger_spans.sort()
            argument_spans.sort()
            links.sort()

            if self._use_gold_triggers and (not trigger_spans) and (not has_gold_links):
                logger.info(f"No triggers given for this document: {file_path}. Skipping...")
                continue

            if self._use_gold_arguments and (not argument_spans) and (not has_gold_links):
                logger.info(f"No arguments given for this document: {file_path}. Skipping...")
                continue

            doc_trigger_arg_info = defaultdict(list)
            for link in links:
                trigger, argument, role = link
                doc_trigger_arg_info[tuple(trigger)].append((tuple(argument), ROLE_MAP(role)))

            if (not skip_if_no_gold_links) and (len(recoverable_links) == 0):
              for trigger in trigger_spans:
                # placeholder dummy argument of minimal length so that things don't crash during prediction mode due to span width hyperparameter settings
                # !!!!! This means that you should not rely on the automatically reported metrics, but rely only on the output of the RAMS scorer
                doc_trigger_arg_info[tuple(trigger)].append((tuple([0,0]), ROLE_MAP("DUMMY_ARGUMENT")))

            genre = input_["doc_key"].split('_')[0]
            doc_id = input_["doc_key"]

            yield self.text_to_instance(sentences,
                                        sentence_start_offsets,
                                        genre,
                                        doc_id,
                                        trigger_spans,
                                        argument_spans,
                                        doc_trigger_arg_info)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        # gather files
        import os
        if os.path.isdir(file_path):
            dir_path = file_path
            dir_contents = os.listdir(dir_path)
            file_paths = [os.path.join(dir_path, f) for f in dir_contents if f.endswith('.json')]
        elif os.path.isfile(file_path):
            file_paths = [file_path]
        else:
            import glob
            file_paths = glob.glob(file_path)  # `file_path` may include wildcard characters

        file_paths = list(filter(lambda x: os.path.basename(x) not in REMOVE, file_paths))

        logger.info("Reading RAMS arglinking instances from dataset files at: %s", file_path)
        return self._read_from_json(file_paths)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentences: List[List[str]],
                         sentence_start_offsets: List[int],
                         genre: str,
                         document_id: str,
                         possible_triggers,  # all given spans, regardless of whether they participate in a link
                         possible_args,  # all given spans, regardless of whether they participate in a link
                         doc_trigger_arg_info: Optional[Dict[Tuple[int, int], List[Tuple[str, Tuple[int, int]]]]] = None) -> Instance:
        # pylint: disable=arguments-differ
        max_sent_len = max(len(s) for s in sentences)
        padded_sentences = [[sentence[i] if i < len(sentence) else "UNK"
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
        assert sentence_offsets == sentence_start_offsets
        sentence_offsets.append(float('inf'))  # sentinel/padding for finding sentence id from span

        text_field = TextField([Token(word) for word in flattened_sentences], self._token_indexers)

        if genre:
            genre_field = LabelField(self._genres[genre], label_namespace="genre_labels", skip_indexing=True)
        else:
            genre_field = LabelField(0, label_namespace="genre_labels", skip_indexing=True)

        # pack up data as: List[str], List[List[SpanField]]
        #                       |               |
        #                    labels      (trig,arg) span pairs
        has_links = bool(doc_trigger_arg_info)
        trigger_arg_pairs = [] if has_links else None
        triggers = []
        args = []
        roles = [] if has_links else None
        trigger_sent_ids = {}
        arg_sent_ids = {}

        for p in possible_triggers:
            trigger_sent_ids[p] = self._get_sentence_id(p, sentence_offsets)
        for a in possible_args:
            arg_sent_ids[a] = self._get_sentence_id(a, sentence_offsets)

        if has_links:
            trigger_arg_pairs = []
            seen_trigger_spans = set()
            seen_arg_spans = set()
            for trigger_span, argument_data in doc_trigger_arg_info.items():
                for (argument_span, role) in argument_data:
                    trigger_arg_pairs.append([trigger_span, argument_span, role])

                    # If not using gold, need to make sure the labels
                    # we care about exist in the enumerated spans
                    span_size_condition = True
                    if (not self._use_gold_triggers):
                        span_size_condition &= (trigger_span[1] - trigger_span[0] < self._max_trigger_span_width)
                    if (not self._use_gold_arguments):
                        span_size_condition &= (argument_span[1] - argument_span[0] < self._max_arg_span_width)
                    if span_size_condition:
                        roles.append(LabelField(role, label_namespace="labels"))
                        triggers.append(trigger_span)
                        args.append(argument_span)

        else:
            # Just get triggers and arguments
            for trigger_span in possible_triggers:
                span_size_condition = True
                if (not self._use_gold_triggers):
                    span_size_condition &= (trigger_span[1] - trigger_span[0] < self._max_trigger_span_width)
                if span_size_condition:
                    triggers.append(trigger_span)
            for argument_span in possible_args:
                span_size_condition = True
                if (not self._use_gold_arguments):
                    span_size_condition &= (argument_span[1] - argument_span[0] < self._max_arg_span_width)
                if span_size_condition:
                    args.append(argument_span)

        metadata: Dict[str, Any] = dict()
        metadata["sentences"] = sentences
        metadata["sentence_start_offsets"] = sentence_start_offsets
        metadata["text_lens"] = np.array(text_lens_mask)
        metadata["doc_id"] = document_id
        metadata["has_gold_targets"] = has_links
        metadata["data_path"] = self._file_path
        metadata["language"] = self._language
        metadata["annotation_kind"] = self._annotation_mode

        # Create all triggers and all args
        if self._use_gold_triggers:
            trigger_spans_field, trigger_idx_field, trigger_sentence_id_field = self._get_unique_spans(triggers, text_field, trigger_sent_ids)
        else:
            trigger_spans_field, trigger_idx_field, trigger_sentence_id_field = self._enumerate_spans(sentences, self._max_trigger_span_width, text_field, triggers)
        if self._use_gold_arguments:
            arg_spans_field, arg_idx_field, arg_sentence_id_field = self._get_unique_spans(args, text_field, arg_sent_ids)
        else:
            #arg_spans_field, arg_idx_field, arg_sentence_id_field = self._enumerate_spans(sentences, self._max_arg_span_width, text_field, args, possible_args=possible_args)  #### EXPERIMENT: USE SYNTACTIC SPANS
            arg_spans_field, arg_idx_field, arg_sentence_id_field = self._enumerate_spans(sentences, self._max_arg_span_width, text_field, args)  #### EXPERIMENT: USE ALL SPANS

        roles_field = ListField(roles) if roles else None

        metadata['triggers'] = triggers
        metadata['arguments'] = args
        if trigger_arg_pairs:
            metadata['roles'] = list(zip(*trigger_arg_pairs))[2]

        f_sentences = [word for sentence in sentences for word in sentence]
        gold_strings = [" ".join(f_sentences[arg[0]:arg[1]+1]) for arg in metadata['arguments']]
        metadata['gold_strings'] = gold_strings

        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {
            "text": text_field,
            "genre": genre_field,
            "metadata": metadata_field,
            "all_triggers": trigger_spans_field,
            "all_args": arg_spans_field,
            "all_trigger_sentence_ids": trigger_sentence_id_field,
            "all_arg_sentence_ids": arg_sentence_id_field,
            "target_trigger_idx": trigger_idx_field,
            "target_arg_idx": arg_idx_field,
            }

        if roles_field:
            fields["target_roles"] = roles_field

        return Instance(fields)

    def _get_sentence_id(self, span, sentence_offsets):
        start = span[0]
        end = span[1]

        for i in range(len(sentence_offsets)-1):
            if (start >= sentence_offsets[i]) and (end < sentence_offsets[i+1]):
                # right-exclusive: sentence_offsets[i+1] marks the index of the first token of sentence i+1
                return i

        return CROSSES_SENTENCE_BOUNDARY

    def _get_unique_spans(self, spans, text, sentence_ids):
        # span_indices are effectively the labels
        spans, span_indices = np.unique(np.array(spans), return_inverse=True, axis=0)
        spans_list_field = ListField([SpanField(int(s[0]), int(s[1]), text) for s in spans])
        labels_index_field = ListField([IndexField(int(idx), spans_list_field) for idx in span_indices])

        sent_ids = [sentence_ids[(s[0], s[1])] for s in spans]
        sentence_id_field = ListField([LabelField(si, label_namespace="sentence_id_tags", skip_indexing=True) for si in sent_ids])

        return spans_list_field, labels_index_field, sentence_id_field

    def _enumerate_spans(self, sentences, max_span_width, text, labels_list, possible_args=None):
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
                if (possible_args is None) or ((start, end) in possible_args):
                  spans.append(SpanField(start, end, text))
                  python_spans.append((start, end))
                  sentence_ids.append(LabelField(sentence_id, label_namespace="sentence_id_tags", skip_indexing=True))
            sentence_offset += len(sentence)
        python_span_dict = {span:i for i, span in enumerate(python_spans)}
        labels_idx = [python_span_dict.get(span, None) for span in labels_list]  # given spans might cross sentence boundaries (e.g., due to parser errors), so remove them to avoid issues in the model
        labels_idx = [i for i in labels_idx if i != None]
        spans_list_field = ListField(spans)
        labels_index_field = ListField([IndexField(idx, spans_list_field) for idx in labels_idx])

        sentence_id_field = ListField(sentence_ids)
        return spans_list_field, labels_index_field, sentence_id_field
