import argparse
import json
import os
import spacy
from spacy.tokens import Doc
from benepar.spacy_plugin import BeneparComponent

# Benepar tag set: ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'S', 'SBAR', 'SBARQ', 'SINV', 'SQ', 'UCP', 'VP', 'WHADJP', 'WHADVP', 'WHNP', 'WHPP', 'X']

filter_fn = filterf = lambda c: any(label in ["NP"] for label in c._.labels)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--documents', type=str, required=True, help='Documents to parse')
  parser.add_argument('--output-dir', type=str, required=True, help='Where to store output')
  
  args = parser.parse_args()
  return args

def extract_spans(parser, nlp, document):
  spans = []
  for sentence_tokens in document:
    # use pre-tokenized text
    doc = Doc(nlp.vocab, words=sentence_tokens, spaces=[True]*(len(sentence_tokens)-1) + [False])
    try:
      doc = custom_sentencizer(doc)
      doc = parser(doc)

      sent = list(doc.sents)[0]
    except:
      try:
        # Benepar has a length limit of 300 tokens (https://github.com/nikitakit/self-attentive-parser/issues/37)
        assert len(doc) == 1 or len(doc) >= 300
      except:
        print(doc)
        import pdb; pdb.set_trace()
      continue

    try:
      extracted_spans = list(filter(filter_fn, sent._.constituents))
    except:
      print("Could not filter")
      import pdb; pdb.set_trace()

    for span in extracted_spans:
      try:
        # the tokenization of the spans should match the pre-tokenization of the document
        assert span.text == " ".join(sentence_tokens[span.start:span.end])
      except:
        print(f"Parsed {span.text}, document says {sentence_tokens[span.start:span.end]}")
        print(list([i, doc[i]] for i in range(len(doc))))
        doc2 = Doc(nlp.vocab, words=sentence_tokens, spaces=[True]*(len(sentence_tokens)-1) + [False])
        print(list([i, doc2[i]] for i in range(len(doc2))))
        import pdb; pdb.set_trace()

      spans.append([span.start, span.end-1, [["UNKNOWN_TYPE", 1.0]]])  # span.end is exclusive in SpaCy, but RAMS spans are inclusive

    print(sent)
    print(extracted_spans)
    print()
  return spans

def get_documents(path):
  if os.path.isdir(path):
    # pointing to directory
    dir_path = path
    dir_contents = os.listdir(dir_path)
    documents = [os.path.join(dir_path, f) for f in dir_contents if f.endswith('.json')]
  elif os.path.isfile(path):
    # pointing to single file
    documents = [path]
  else:
    # pointing to file glob (e.g., includes wildcard characters)
    import glob
    documents = glob.glob(path)

  return documents

def custom_sentencizer(doc):
  for i, token in enumerate(doc):
    if i == 0:
      doc[i].is_sent_start = True
    else:
      doc[i].is_sent_start = False
  return doc

def main():
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU

  args = parse_args()

  nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])
  ConstituencyParser = BeneparComponent("benepar_en2")

  documents = get_documents(args.documents)
  
  for document in documents:
    with open(document, "r") as f:
      data = json.load(f)

    sentences = data["sentences"]

    gold_spans = data["ent_spans"]
    gold_span_boundaries = [(span[0], span[1]) for span in gold_spans]

    syntactic_spans = extract_spans(ConstituencyParser, nlp, sentences)
    syntactic_spans = [span for span in syntactic_spans if (span[0], span[1]) not in gold_span_boundaries]  # deduplicate
    spans = syntactic_spans + gold_spans

    data["ent_spans"] = spans

    # write out new spans to copy of document (don't overwrite)    
    out_document = os.path.join(args.output_dir, os.path.basename(document))
    with open(out_document, "w") as f:
      json.dump(data, f, indent=4)

if __name__ == "__main__":
  main()
