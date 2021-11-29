These files were used for to generate the cached lm embeddings for all results
in the paper.

## Usage

There are now better ways to do this with the Transformers library without
installing another dependency (Tensorflow).

To tokenize the RAMS text for use in BERT (`vocab.txt` is BERT's vocab), run

```python minimize_json.py vocab.txt <dir_containing_jsonlines_file> <output_dir> false```

The `output_dir` will contain 4 files based on the length of a segment. We 
only care about the "512" ones (e.g. `*.512.jsonlines`). Concatenate all of 
these files (e.g. `cat *.512.jsonlines >> dev.512.jsonlines`) to get a 
version that can then be cached with `cache_bert_hf.py`.

## Other

More generally, to use another encoder and to do this without installing
dependencies, you can reference this [similar script](https://github.com/pitrack/incremental-coref/blob/main/conversion_scripts/minimize_json.py#L186)
that I wrote for a different project that instead uses the Transformers library.