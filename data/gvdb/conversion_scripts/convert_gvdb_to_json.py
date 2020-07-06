import csv
import sys
import json

def convert_gvdb_to_span(gvdb_dict, role):
  if gvdb_dict['startIndex'] != -1:
    # +2 is for the ". " added to separate title and body
    return ((int(gvdb_dict['startIndex']) + 2,
             int(gvdb_dict['endIndex']) + 2,
             role,
             gvdb_dict['value']))
  else:
    return None

cgts = convert_gvdb_to_span

def convert_json_to_char_spans(blob, full_text, tok_map, tokenized):
  # circumstances
  all_roles = []
  all_roles.append(cgts(blob['circumstances']['number-of-shots-fired'], "CIR-NUM-SHOTS"))
  all_roles.append(cgts(blob['circumstances']['type-of-gun'], "CIR-WEAPON"))
  all_roles.append(cgts(blob['date-and-time']['city'], "DAT-CITY"))
  all_roles.append(cgts(blob['date-and-time']['clock-time'], "DAT-CLOCK"))  
  all_roles.append(cgts(blob['date-and-time']['details'], "DAT-LOC"))
  all_roles.append(cgts(blob['date-and-time']['time-day'], "DAT-TIME"))
  for victim in blob['victim-section']:
    all_roles.append(cgts(victim['age'], "VIC-AGE"))
    all_roles.append(cgts(victim['name'], "VIC-NAME"))
    all_roles.append(cgts(victim['race'], "VIC-RACE"))
    break # only look at first one
  for shooter in blob['shooter-section']:
    all_roles.append(cgts(shooter['age'], "SHO-AGE"))
    all_roles.append(cgts(shooter['name'], "SHO-NAME"))
    all_roles.append(cgts(shooter['race'], "SHO-RACE"))
    break # only look at first one
  
  all_roles = [role for role in all_roles if role is not None]
  checksums = [role[3] == full_text[min(role[0], role[1]):max(role[0], role[1])]
               for role in all_roles]
  checksums_2 = [role[3] == full_text[min(role[0], role[1]) - 2:max(role[0], role[1]) - 2]
                for role in all_roles]
  for i, (isVerified, isVerified2) in enumerate(zip(checksums, checksums_2)):
    if not isVerified and not isVerified2:
      old_start = all_roles[i][0]
      old_end = all_roles[i][1]
      old_role = all_roles[i][2]
      old_value = all_roles[i][3].strip()
      start = full_text.find(old_value)
      if start == -1 or old_value == "":
        print ("LOST:\t\t {} [==x]".format(all_roles[i]))
        all_roles[i] = None
      else:
        all_roles[i] = (start, start + len(old_value),
                        old_role, old_value)
        print ("NO:\t\t{} [==>] {}".format(full_text[old_start:old_end], all_roles[i]))
    elif not isVerified:
      new_start = all_roles[i][0] - 2
      new_end = all_roles[i][1] - 2
      all_roles[i] = (new_start, new_end,
                      all_roles[i][2], all_roles[i][3])
      print ("MAYBE:\t\t{} [=?=] {}".format(full_text[new_start:new_end],
                                            all_roles[i]))
    else:      
      print ("YES:\t\t{} [===] {}".format(full_text[all_roles[i][0]:all_roles[i][1]],
                                          all_roles[i]))

  for i, span in enumerate(all_roles):
    if span is None:
      continue
    old_start = span[0]
    old_end = span[1]
    old_role = span[2]
    old_value = span[3]
    if old_value == "":
      all_roles[i] = None
    elif old_value[0] == " ":
      all_roles[i] = (old_start + 1, old_end,
                      old_role, old_value)
  # retokenize
  def retokenize(span):
    start_idx = int(tok_map[span[0]])
    # If you hit an exception here, check the original data, maybe the document is empty?
    end_idx = int(tok_map[span[1] - 1]) + 1
    return (start_idx, end_idx, span[2], span[3], tokenized[start_idx:end_idx])
    
  return [retokenize(role) for role in all_roles if role is not None]

from spacy.lang.en import English
import numpy as np

nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))

def process_full_text(text):
  text = text.replace('\x96', '-').replace('\x97', '-')
  tokenized_text = nlp(text)
  sents = list(tokenized_text.sents)
  sentences = [[w.text for w in s] for s in sents]
  flat_char_offs = [w.idx for s in sents for w in s] + [len(text)]
  flat_token_offs = [w.i for s in sents for w in s]
  diffs = list(np.diff(flat_char_offs))
  char_tok_map = np.repeat(flat_token_offs, diffs)
  continuous = [w.text for s in sents for w in s]
  return sentences, continuous, char_tok_map

def process_file(file_it, doc_dict, avoid_dict):
  num_spans = 0
  next(file_it, None)
  for i, row in enumerate(file_it):
    title = row[-4]
    text_body = row[-3].split(title)[-1]
    full_text = title.strip() + ". " + text_body.strip()
    json_blob = json.loads(row[-2])
    new_json = {}
    new_json["doc_key"] = str(i)
    new_json["date"] = json_blob["date-and-time"]["date"]
    if len(new_json["date"]) == 0 or str(new_json["date"])[0] == "{":
      continue
    new_json["full_text"], continuous, tok_map = process_full_text(full_text)
    new_json["spans"] = convert_json_to_char_spans(json_blob, full_text, tok_map, continuous)

    if full_text in doc_dict or full_text in avoid_dict:
      continue
    else:
      doc_dict[full_text] = new_json
    num_spans += len(new_json["spans"])
  return num_spans

test_file = open(sys.argv[2], 'r', encoding='utf-8')
in_file = open(sys.argv[1], 'r', encoding='utf-8')
tsvtest = csv.reader(test_file, delimiter='\t')
tsvin = csv.reader(in_file, delimiter='\t')

test_docs = {}
train_docs = {}

num_test_spans = process_file(tsvtest, test_docs, {})

chron_test_spans = sorted(list(test_docs.values()), key=lambda x:str(x["date"]))
filtered = chron_test_spans
filtered_date = [x["date"] for x in filtered]
filtered_spans = [x["spans"] for x in filtered]

test = open("test.json", 'w+', encoding='utf-8')
dev = open("dev.json", 'w+', encoding='utf-8')
train = open("train.json", 'w+', encoding='utf-8')

for json_blob in filtered[:-1010]:
   train.write(json.dumps(json_blob))
   train.write("\n")

for json_blob in filtered[-1010:-610]:
   dev.write(json.dumps(json_blob))
   dev.write("\n")

for json_blob in filtered[-510:-10]:
   test.write(json.dumps(json_blob))
   test.write("\n")

# The top 10 have clearly incorrect dates, like something in the year 3000
