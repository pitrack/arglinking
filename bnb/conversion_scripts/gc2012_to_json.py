import json
import sys
import xml.etree.ElementTree as ET

from nltk.corpus import ptb
from nltk.tree import ParentedTree

from typing import Dict, List, Tuple, Iterable, Union, Any, Optional

from collections import defaultdict


def valid_token(ptree, token_idx):
    position = ptree.leaf_treeposition(token_idx)
    parent_position = position[:-1]
    parent_label = ptree[parent_position].label()
    return parent_label != '-NONE-'

def index_tree(tree, offset):
    """
    Keep track of token indices.
    See https://stackoverflow.com/questions/36831354/absolute-position-of-leaves-in-nltk-tree/54735452#54735452
    """
    for idx, _ in enumerate(tree.leaves()):
        tree_location = tree.leaf_treeposition(idx)
        non_terminal = tree[tree_location[:-1]]
        # non_terminal.append(idx)
        non_terminal[0] = (non_terminal[0], idx+offset)
    return tree

def parse_node_id(node):
    assert node is not None
    assert node[:3] == 'wsj'
    id = node.split('_')[-1]  # xxxx:a:b:c

    article_id, sentence_id, head_token_id, tree_height = id.split(':')
    sentence_id = int(sentence_id)
    head_token_id = int(head_token_id)
    tree_height = int(tree_height)
    section_id = article_id[:2]  # article `wsj_{article_id}.mrg` is listed under `.../parsed/mrg/wsj/{section_id}`
    return section_id, article_id, sentence_id, head_token_id, tree_height

def node2span(node, offsets):
    section_id, article_id, sentence_id, head_token_id, tree_height = parse_node_id(node)

    ptree = ParentedTree.convert(ptb.parsed_sents(f"wsj/{section_id}/wsj_{article_id}.mrg")[sentence_id])

    # Index each leaf node with its offset into the document
    offset = offsets[article_id][sentence_id]
    ptree = index_tree(ptree, offset)

    leaf_position = ptree.leaf_treeposition(head_token_id)
    span_position = leaf_position[:-(tree_height+1)]
    span_tokens = ptree[span_position].leaves()

    return section_id, article_id, sentence_id, span_tokens, ptree

def is_consecutive_and_sorted(l):
    return l == list(range(min(l), max(l)+1))

def indices2range(ii):
    tokens = [x[0] for x in ii]
    indices = [x[1] for x in ii]
    if not is_consecutive_and_sorted(indices):
        import pdb; pdb.set_trace()

    return tokens, (min(indices), max(indices))

def _read_document(file_path: str):
    print(f"Reading GC2012 instances from dataset file at: {file_path}")

    xml_tree = ET.parse(file_path)
    root = xml_tree.getroot()

    # Read in all relevant documents to get token offsets
    sentence_offsets = dict()  # sentence_offsets[doc][sentence] = starting_token_idx

    # Remove special parse tokens (e.g., "*RNR*-1") from text and reindex tokens
    # It appears that all special characters have a parent of '-NONE-'.
    token_map = dict()  # token_map[doc][original_token_idx] = remapped_token_idx
    texts = dict()
    filtered_texts = dict()
    for annotations in root.getchildren():
        trigger_node = annotations.get('for_node')  # wsj_xxxx:a:b:c
        section_id, article_id, _, _, _ = parse_node_id(trigger_node)

        if article_id in sentence_offsets.keys():
            # we've already processed this document
            continue
        else:
            sentence_offsets[article_id] = dict()
            token_map[article_id] = dict()

        parse_trees = ptb.parsed_sents(f"wsj/{section_id}/wsj_{article_id}.mrg")
        total_valid_tokens_seen = 0
        total_tokens_seen = 0
        text = []
        filtered_text = []
        for sent_id, parse_tree in enumerate(parse_trees):
            ptree = ParentedTree.convert(parse_tree)

            tokens = ptree.leaves()
            valid_token_indices = [i if valid_token(ptree, i) else None for i,x in enumerate(tokens)]
            valid_tokens = [x for i,x in enumerate(tokens) if valid_token(ptree, i)]

            sentence_offsets[article_id][sent_id] = total_tokens_seen
            for i,x in enumerate(valid_token_indices):
                if x is None:
                    # special token that should be removed (e.g. *RNR*-1)
                    token_map[article_id][total_tokens_seen] = None
                else:
                    token_map[article_id][total_tokens_seen] = total_valid_tokens_seen
                    total_valid_tokens_seen += 1

                total_tokens_seen += 1


            text.append(tokens)
            filtered_text.append(valid_tokens)

        texts[article_id] = text
        filtered_texts[article_id] = filtered_text


    # See `http://lair.cse.msu.edu/projects/implicit_annotations.html` for details.
    packets = []
    for annotations in root.getchildren():
        trigger_node = annotations.get('for_node')  # wsj_xxxx:a:b:c
        trigger_section_id, trigger_article_id, trigger_sentence_id, trigger_span_tokens, _ = node2span(trigger_node, sentence_offsets)

        # Readjust token indices since we removed special tokens (this probably doesn't happen in the data, but just to be safe)
        trigger_text, original_trigger_span = indices2range(trigger_span_tokens)
        trigger_span = (token_map[trigger_article_id][original_trigger_span[0]], token_map[trigger_article_id][original_trigger_span[1]])

        packet = {"document_id": f"wsj_{trigger_article_id}",
                  "document": filtered_texts[trigger_article_id],  # filtered document (does not include special parse tokens)
                  "trigger": {"node_id": trigger_node,
                              "span": trigger_span,  # offset into filtered document
                              "text": trigger_text},
                  "arguments": defaultdict(list)}

        printed_trigger = False
        for annotation in annotations.getchildren():
            argument_node = annotation.attrib.get('node')
            argument_section_id, argument_article_id, argument_sentence_id, argument_span_tokens, _ = node2span(argument_node, sentence_offsets)

            if trigger_section_id != argument_section_id:
                raise ValueError(f"Trigger and argument should be in same section: got trigger_section_id={trigger_section_id}, argument_section_id={argument_section_id}")
            if trigger_article_id != argument_article_id:
                raise ValueError(f"Trigger and argument should be in same article: got trigger_article_id={trigger_article_id}, argument_article_id={argument_article_id}")

            argn = annotation.attrib.get('value')

            # get `attribute`
            assert len(annotation.getchildren()) == 1
            assert annotation.getchildren()[0].tag == 'attributes'

            assert len(annotation.getchildren()[0].getchildren()) <= 1
            if len(annotation.getchildren()[0].getchildren()) == 1:
                attribute = annotation.getchildren()[0].getchildren()[0].text
            else:
                attribute = ""


            # Readjust token indices since we removed special tokens
            argument_text, original_argument_span = indices2range(argument_span_tokens)
            argument_span = (token_map[argument_article_id][original_argument_span[0]], token_map[argument_article_id][original_argument_span[1]])

            if attribute == "Split":
                if not printed_trigger:
                    print("Trigger", trigger_node, trigger_sentence_id, trigger_text, trigger_span)
                    printed_trigger = True
                print(argn, attribute, argument_node, argument_sentence_id, argument_text, argument_span)

            packet["arguments"][argn].append({"node_id": argument_node,
                                              "span": argument_span,  # offset into filtered document
                                              "attribute": attribute,
                                              "text": argument_text})


        if printed_trigger:
            print(packet)
            print()


        packets.append(packet)

    return packets

def main():
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    packets = _read_document(input_file_path)

    with open(output_file_path, "w") as f:
        json.dump(packets, f)


if __name__ == "__main__":
    main()
