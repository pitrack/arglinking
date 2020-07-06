import os
import argparse

from miso.utils.params import Params
from miso.utils import logging
from miso.data.iterators import BucketIterator, BasicIterator
from miso.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from miso.data.dataset_readers import RAMSDatasetReader
from miso.data.dataset_readers import GVDBDatasetReader

logger = logging.init_logger()

def load_dataset_reader(dataset_type, *args, **kwargs):
    if dataset_type == "RAMS":
        dataset_reader = RAMSDatasetReader(
            max_trigger_span_width=kwargs.get('max_trigger_span_width'),
            max_arg_span_width=kwargs.get('max_arg_span_width'),
            use_gold_triggers=kwargs.get('use_gold_triggers'),
            use_gold_arguments=kwargs.get('use_gold_arguments'),
            annotation_mode=kwargs.get('annotation_mode'),
            language=kwargs.get('language'),
            genres=kwargs.get('genres'),
            token_indexers=dict(
                tokens=SingleIdTokenIndexer(namespace='tokens'),
                token_characters=TokenCharactersIndexer(namespace='characters')
            )
        )
    elif dataset_type == "GVDB":
        dataset_reader = GVDBDatasetReader(
            max_trigger_span_width=kwargs.get('max_trigger_span_width'),
            max_arg_span_width=kwargs.get('max_arg_span_width'),
            use_gold_triggers=kwargs.get('use_gold_triggers'),
            use_gold_arguments=kwargs.get('use_gold_arguments'),
            token_indexers=dict(
                tokens=SingleIdTokenIndexer(namespace='tokens'),
                token_characters=TokenCharactersIndexer(namespace='characters')
            )
        )

    return dataset_reader

def load_dataset(path, dataset_type, *args, **kwargs):
    return load_dataset_reader(dataset_type, *args, **kwargs).read(path)

def dataset_from_params(params):
    train_data = os.path.join(params['data_dir'], params['train_data'])
    dev_data = os.path.join(params['data_dir'], params['dev_data'])
    test_data = params['test_data']
    data_type = params['data_type']

    logger.info("Building train datasets ...")
    train_data = load_dataset(train_data, data_type, **params)

    logger.info("Building dev datasets ...")
    dev_data = load_dataset(dev_data, data_type, **params)

    if test_data:
        test_data = os.path.join(params['data_dir'], params['test_data'])
        logger.info("Building test datasets ...")
        test_data = load_dataset(test_data, data_type, **params)

    return dict(
        train=train_data,
        dev=dev_data,
        test=test_data
    )


def iterator_from_params(vocab, params):
    iter_type = params['iter_type']
    train_batch_size = params['train_batch_size']
    test_batch_size = params['test_batch_size']
    max_instances = params.get('max_instances_in_memory', None)
    instances_per_epoch = params.get('instances_per_epoch', None)

    if iter_type == "BucketIterator":
        train_iterator = BucketIterator(
            sorting_keys=list(map(tuple, params.get('sorting_keys', []))),
            batch_size=train_batch_size,
            padding_noise = float(params.get('padding_noise', 0.1))
        )
    elif iter_type == "BasicIterator":
        train_iterator = BasicIterator(
            batch_size=train_batch_size,
            max_instances_in_memory=max_instances,
            instances_per_epoch=instances_per_epoch
        )
    else:
        raise NotImplementedError

    dev_iterator = BasicIterator(
        batch_size=train_batch_size
    )

    test_iterator = BasicIterator(
        batch_size=test_batch_size
    )

    train_iterator.index_with(vocab)
    dev_iterator.index_with(vocab)
    test_iterator.index_with(vocab)

    return train_iterator, dev_iterator, test_iterator
