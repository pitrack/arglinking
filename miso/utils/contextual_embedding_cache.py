import numpy as np

SENTENCE_CUTOFF = float('inf')
def load_lm_embeddings(doc_key, lm_file, lm_size, lm_layers):
  if lm_file is None:
    return np.zeros([0, 0, lm_size, len(lm_layers)])
  file_key = doc_key.replace("/", ":")
  group = lm_file[file_key]
  num_sentences = min(len(list(group.keys())), SENTENCE_CUTOFF)
  sentences = [group[str(i)][...] for i in range(num_sentences)]
  try:
    lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), lm_size, len(lm_layers)])
  except Exception as e:
    print ("[{}, {}, {}, {}]".format(num_sentences,
                                     max(s.shape[0] for s in sentences),
                                     lm_size, len(lm_layers)))
    raise e
  for i, s in enumerate(sentences):
    lm_emb[i, :s.shape[0], :, :] = s[:, :, lm_layers]
  return lm_emb
