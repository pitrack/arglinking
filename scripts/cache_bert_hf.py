"""
To use this script:

python cache_bert_hf.py file1.jsonlines file2.jsonlines ... filen.jsonlines output_cache.hdf5

Each file1.jsonlines should contain:
-- "sentences": a list of list of subtokens
-- "doc_key": a unique id
-- "subtoken_map": a map from subtoken index to token index
-- "sentence_map": a map from subtoken index to sentence index

See the output of https://github.com/mandarjoshi90/coref/blob/master/minimize.py for more details.

You may want to run minimize.py first or skip this script entirely, depending on your use case.
"""
import torch
import h5py
import json
import sys
import time
from transformers import *

# CACHED_LAYERS = [6, 7, 8, 9]
CACHED_LAYERS = [9, 10, 11, 12]

# Transformers has a unified API
# for 8 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
# MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
#           (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
#           (GPT2Model,       GPT2Tokenizer,       'gpt2'),
#           (CTRLModel,       CTRLTokenizer,       'ctrl'),
#           (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
#           (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
#           (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
#           (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
#           (RobertaModel,    RobertaTokenizer,    'roberta-base')]

# To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

# Let's encode some text in a sequence of hidden-states using each model:

class CachingEncoder():
  def __init__(self):
    # Static
    model_class = BertModel
    tokenizer_class = BertTokenizer
    pretrained_weights = 'bert-base-cased'
    # Exposed
    self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    self.model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)

  def tokenize_subtokens(self, subtokens):
    return [self.tokenizer._convert_token_to_id(subtoken) for subtoken in subtokens]

  def encode(self, text, token_map=[], full_tokenize=False):
    # Encode text
    if full_tokenize:
      subtokens = self.tokenizer.encode(text)
    else:
      subtokens = self.tokenize_subtokens(text)
    input_ids = torch.tensor([subtokens])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    # (num_layers * [num_seqs * seq_len * emb_size])
    with torch.no_grad():
      hidden_states = self.model(input_ids)[-1]  # Models outputs are now tuples, i.e. a 13-tuple for BERT
    if token_map:
      all_layers = []
      for layer in CACHED_LAYERS:
        sequence = hidden_states[layer]
        all_sequences = []
        for _, sequence in enumerate(sequence):
          collected = []
          current_embs = []
          for timestep, emb in enumerate(sequence.unbind()):
            current_marker = max(0, token_map[timestep])
            if current_marker == len(collected):
              # add to current_embs
              current_embs.append(emb)
            else:
              # clear out existing current_embs
              if len(current_embs) == 0:
                print (timestep, token_map)
              collected_emb = torch.mean(torch.stack(current_embs),
                                         0)
              collected.append(collected_emb)
              current_embs = []
              # add to current_embs
              current_embs.append(emb)
          # clear out the last current_embs
          collected_emb = torch.mean(torch.stack(current_embs),
                                     0)
          collected.append(collected_emb)
          # append this layer to all layers
          all_sequences.append(torch.stack(collected))
        all_layers.append(torch.stack(all_sequences))
      hidden_states = all_layers
    return hidden_states

def cache_file(data_paths, out_file_name, caching_encoder):
  start_time = time.time()
  with h5py.File(out_file_name, "w") as out_file:
    for data_path in data_paths:
      with open(data_path) as in_file:
        for doc_num, line in enumerate(in_file):
          example = json.loads(line)
          sentences = example["sentences"]
          file_key = example["doc_key"].replace("/", ":")
          subtoken_map = example["subtoken_map"]
          sentence_map = example["sentence_map"]
          sentence_dict = {}
          for (subtoken_idx, sentence_idx) in zip(subtoken_map, sentence_map):
            if subtoken_idx not in sentence_dict:
              sentence_dict[subtoken_idx] = sentence_idx
          num_sentences = max(sentence_map)
          # print (num_sentences)
          encoder_output_by_sentence = [[] for _ in range(num_sentences)]
          try:
            group = out_file.create_group(file_key)
          except:
            print ("Skipping {}".format(file_key))
            continue
          segment_offset = 0
          segment_token_map = subtoken_map
          for j, segment in enumerate(sentences):
            # print (j, segment, segment_token_map)
            encoder_output_by_layers = []
            # (num_layers * [num_sequences * seq_length * emb_size]), num_sequences=1
            encoder_output_torch = caching_encoder.encode(segment,
                                                          token_map=segment_token_map)
            # ([seq_length * emb_size * num_layers],)
            encoder_output_reshaped = torch.stack(encoder_output_torch).permute(1,2,3,0).unbind()
            assert len(encoder_output_reshaped) == 1
            # (seq_length * [emb_shape * num_layers])
            encoder_token_output = encoder_output_reshaped[0].unbind()
            for i, token in enumerate(encoder_token_output):
              sentence_idx = sentence_dict[i + segment_offset]
              # print (sentence_idx, segment_offset, i)
              encoder_output_by_sentence[sentence_idx].append(token)
            num_tokens = len(encoder_token_output)
            segment_offset += num_tokens
            segment_token_map = [index - num_tokens
                                 for index in segment_token_map[len(segment):]]
            # group[sent] = seq_length * emb_size * num_layer
          for i, sentence in enumerate(encoder_output_by_sentence):
            group[str(i)] = torch.stack(sentence) if sentence else torch.zeros([0,768,13])
          if doc_num % 1 == 0:
            print("[{:.2f}s] Cached {} documents in {}".format(
              time.time() - start_time,
              doc_num + 1,
              data_path))


if __name__ == "__main__":
  encoder = CachingEncoder()
  cache_file(sys.argv[1:-1], sys.argv[-1], encoder)
