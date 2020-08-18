# Beyond NomBank (BNB)

## About

This directory contains the version of BNB used in [Multi-Sentence Argument Linking](https://arxiv.org/abs/1911.03766).
It is originally derived from [Gerber and Chai, 2012](https://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00110), available for download [here](http://lair.cse.msu.edu/projects/implicit_annotations.html).
These splits are released with permission from the author.

The `splits` directory contains:

   * `{train, dev, test}_bleached.json`, which are the files used for our experiments except the tokens have been replaced with `<unk>` due to Penn Treebank licensing considerations.

We also provide a `conversion_scripts` directory containing two files:

   * `gc2012_to_json.py` processes the original xml file to json and retrieves the documents' text. There are annotations in the output file of `gc2012_to_json.py`, but they should be ignored because they have not been adjusted to fit within the argument linking framework. The annotations in `{train, dev, test}_bleached.json` are the ones used in the paper. (Adjustment of the annotations, including manual adjustment, was done using separate scripts.)
   * `unbleach.py` re-populates the tokens in a bleached data split (e.g., `train_bleached.json`) using the output of `gc2012_to_json.py`.

## Data Generation

To generate the BNB data used in Multi-Sentence Argument Linking, follow the steps below. Note that you must have a copy of the Penn Treebank because that is the corpus on which the Beyond NomBank annotations are based.

1. Download `implicit_argument_annotations.xml` from [here](http://lair.cse.msu.edu/projects/implicit_annotations.html)

2. Load in the Penn Treebank for NLTK using the instructions [here](http://www.nltk.org/howto/corpus.html#parsed-corpora)

3. Run `python gc2012_to_json.py <path/to/implicit_argument_annotations.xml> documents.json` to get the text of the documents (along with other information you can ignore)

4. Run `python unbleach.py --bleached train_bleached.json --documents documents.json --outfile train_unbleached.json`

5. Run `python unbleach.py --bleached dev_bleached.json --documents documents.json --outfile dev_unbleached.json`

6. Run `python unbleach.py --bleached test_bleached.json --documents documents.json --outfile test_unbleached.json`

## Replication

We found a bug that caused our reported results to be based on greedy decoding instead of argmax decoding. In addition to providing replication results below, we also report results with argmax decoding. The reported metric is F1.

Using BERT layers 9--12:

|                | greedy (dev/test) | argmax (dev/test) |
|----------------|-------------------|-------------------|
| greedy (train) | 73.5/73.8         | 91.5/90.9         |
| argmax (train) | 72.5/73.0         | 89.3/86.5         |

and using BERT layers 6--9:

|                | greedy (dev/test) | argmax (dev/test) |
|----------------|-------------------|-------------------|
| greedy (train) | 72.6/75.1         | 89.8/89.2         |
| argmax (train) | 72.1/75.9         | 91.3/91.0         |

## Citation

If you use wish to reference this dataset, please cite the original dataset release paper. If you use the splits or the json files, please also cite our work.

```
@article{gerber-chai-2012-semantic,
  title = "Semantic Role Labeling of Implicit Arguments for Nominal Predicates",
  author = "Gerber, Matthew  and
    Chai, Joyce Y.",
  journal = "Computational Linguistics",
  volume = "38",
  number = "4",
  year = "2012",
  url = "https://www.aclweb.org/anthology/J12-4003",
  doi = "10.1162/COLI_a_00110",
  pages = "755--798",
}

@inproceedings{ebner-etal-2020-multi,
  title={Multi-Sentence Argument Linking},
  author={Seth Ebner and Patrick Xia and Ryan Culkin and Kyle Rawlins and Benjamin {Van Durme}},
  year={2020},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
}
```