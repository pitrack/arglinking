import sys
import json

dev = "splits/dev.json"
test = "splits/test.json"
train = "splits/train.json"

docs = [dev, test, train]
original = sys.argv[1]

def get_ids(doc):
  doc_ids = []
  with open(doc, 'r') as examples:
    for line in examples:
      doc_ids.append(int(json.loads(line)["doc_key"]))
  return doc_ids

ids = [get_ids(doc) for doc in docs]


in_file = open(original, 'r')
header = next(in_file)
out_dev = [header]
out_test = [header]
out_train = [header]

out_docs = [out_dev, out_test, out_train]
names = ["dev", "test", "train"]
for i, line in enumerate(in_file):
  for split_idx, split in enumerate(ids):
    if i in split:
      out_docs[split_idx].append(line)

for i, doc in enumerate(out_docs):
  out_file = open("splits/{}.tsv".format(names[i]), 'w+')
  out_file.writelines(doc)

