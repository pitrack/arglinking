import argparse
import json

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--bleached', type=str, required=True, help='input bleached data file')
  parser.add_argument('--documents', type=str, required=True, help='input data file containing unbleached documents')
  parser.add_argument('--outfile', type=str, required=True, help='output unbleached data file')

  args = parser.parse_args()

  return args

def unbleach(annotation, text):
  assert [len(s) for s in annotation['sentences']] == [len(s) for s in text]

  flattened_text = [token for sentence in text for token in sentence]

  # Unbleach document text
  annotation['sentences'] = text

  # Unbleach trigger text
  trigger_s, trigger_e = annotation['trigger']['span']  # inclusive index
  annotation['trigger']['text'] = flattened_text[trigger_s:trigger_e+1]

  # Unbleach argument text
  for argument_id in annotation['arguments'].keys():
    for i,argument_mention in enumerate(annotation['arguments'][argument_id]):
      argument_s, argument_e = argument_mention['span']  # inclusive index
      annotation['arguments'][argument_id][i]['text'] = flattened_text[argument_s:argument_e+1]

  return annotation

def main():
  args = parse_args()

  with open(args.documents, 'r') as f:
    documents = json.load(f)
  id2text = dict()
  for document in documents:
    id2text[document['document_id']] = document['document']

  annotations = []
  with open(args.bleached, 'r') as f:
    for line in f:
      annotations.append(json.loads(line))

  unbleached_data = []
  for annotation in annotations:
    unbleached_annotation = unbleach(annotation, id2text[annotation['document_id']])
    unbleached_data.append(unbleached_annotation)

  with open(args.outfile, 'w') as f:
    for item in unbleached_data:
      json.dump(item, f)
      f.write('\n')

if __name__ == "__main__":
  main()
