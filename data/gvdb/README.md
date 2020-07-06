# Gun Violence Database (GVDB)

## About

This directory contains the version of GVDB used in [Multi-Sentence Argument Linking](https://arxiv.org/abs/1911.03766).
It is originally derived from [Pavlick et al., 2016](https://www.aclweb.org/anthology/D16-1106.pdf), available for download [here](http://gun-violence.org/).
These splits are rereleased with permission from the author.

The `splits` directory contains:

   * `{train, dev, test}.json`, which were used for our experiments
   * `{train, dev, test}.tsv`, which contains the raw data but split into time-disjoint chunks.

We also provide a `conversion_scripts` directory containing three files:

   * `readme.txt` describes the original data format (the tsv that is available for download), along with a description of each of the fields. We **do not** include the original files mentioned in the readme.txt in this repo.
   * `convert_gvdb_to_json.py` is a single script which preprocesses the original tsv file to json splits for train, dev, test, each tokenized and with some argument fields populated.
   * `only_split.py` uses the json and original file to re-derive just the raw splits for train, dev, test.

This is a bit backwards: we would expect to first split teh data, then preprocess the splits, and finally convert them to json.
For our purposes, we did all three in one step, and so the scripts we provide here was the easiest way of applying those splits to the raw tsv file.
This is made available for future research that may not want to apply our preprocessing or restrictions on number of arguments per field, but may want to re-use our splits.
These scripts were intended to be run once on a deduped version of the original GVDB dataset.
Please let us (Patrick) know if you want to use or modify these scripts.

## Citation

If you use wish to reference this dataset, please cite the original dataset release paper. If you use the splits or the json files, please also cite our work.

```
@inproceedings{pavlick-etal-2016-gun,
    title = "The Gun Violence Database: A new task and data set for {NLP}",
    author = "Pavlick, Ellie  and
      Ji, Heng  and
      Pan, Xiaoman  and
      Callison-Burch, Chris",
    year = "2016",
    booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
}

@inproceedings{ebner-etal-2020-multi,
  title={Multi-Sentence Argument Linking},
  author={Seth Ebner and Patrick Xia and Ryan Culkin and Kyle Rawlins and Benjamin {Van Durme}},
  year={2020},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
}
```


