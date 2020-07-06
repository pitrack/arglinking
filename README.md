# arglinking

Repository for ACL 2020 paper: [Multi-Sentence Argument Linking](https://arxiv.org/abs/1911.03766) [[pdf]](https://arxiv.org/pdf/1911.03766.pdf).

# Code


This repository uses a combination of files from the MISO repository used internally by Benjamin Van Durme's lab and components from `allennlp`. MISO is largely a fork of `allennlp`.

This code has been verified to run with python 3.6.9 and the packages listed in `requirements.txt`.

## How to train models


For RAMS and GVDB:
```
CUDA_VISIBLE_DEVICES=-1
python -u -m miso.commands.train <path/to/params.yaml>
```

The params file for the RAMS model is located at `params/rams/gold_triggers+gold_arguments/rams.yaml`. It contains `gpu` and `cuda_device` fields that you can modify depending on whether you are using CPU or GPU.

Various experimental settings from Table 2 may be set in the data reader (`miso/data/dataset_readers/rams.py`) and model (`miso/models/arglinking.py`) as designated with comments containing `!!!!!` (caveats/important items to be aware of), `baseline` (trigger representation baselines), and `EXPERIMENT` (use of distractor spans vs. full enumeration of spans).

## How to evaluate models


For RAMS and GVDB:
```
MODEL_DIR=<path/to/model>
OUTPUT_DIR=${MODEL_DIR}/results/test

CUDA_VISIBLE_DEVICES=-1
python -u -m miso.commands.evaluate ${MODEL_DIR}/model.tar.gz
       --weights-file ${MODEL_DIR}/best.th
       --input-file <path to test/*.json>
       --embeddingfile <path to/cached_bert_9_12.hdf5>
       --outputdir ${OUTPUT_DIR}
       --output-file ${OUTPUT_DIR}/test_metrics.json
       --batch-size 1
```

The suffix `9_12` in `cached_bert_9_12.hdf5` refers to using BERT layers 9--12 (the top four layers).

You may also need to set `set -f` to disable wildcard/glob expansion.

# Data

The RAMS dataset is available at https://nlp.jhu.edu/rams/. More information about the data is available on that webpage.

We plan to release the other datasets or splits used in the paper at a later date.

# Citation

If you wish to cite our work:

```
@inproceedings{ebner-etal-2020-multi,
  title={Multi-Sentence Argument Linking},
  author={Seth Ebner and Patrick Xia and Ryan Culkin and Kyle Rawlins and Benjamin {Van Durme}},
  year={2020},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
}
```
