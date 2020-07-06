from typing import Callable, List, Set, Tuple, TypeVar, Optional
import warnings


from miso.utils.checks import ConfigurationError
from miso.data.tokenizers.token import Token


T = TypeVar("T", str, Token)
def enumerate_spans(sentence: List[T],
                    offset: int = 0,
                    max_span_width: int = None,
                    min_span_width: int = 1,
                    filter_function: Callable[[List[T]], bool] = None) -> List[Tuple[int, int]]:
    """
    Given a sentence, return all token spans within the sentence. Spans are `inclusive`.
    Additionally, you can provide a maximum and minimum span width, which will be used
    to exclude spans outside of this range.

    Finally, you can provide a function mapping ``List[T] -> bool``, which will
    be applied to every span to decide whether that span should be included. This
    allows filtering by length, regex matches, pos tags or any Spacy ``Token``
    attributes, for example.

    Parameters
    ----------
    sentence : ``List[T]``, required.
        The sentence to generate spans for. The type is generic, as this function
        can be used with strings, or Spacy ``Tokens`` or other sequences.
    offset : ``int``, optional (default = 0)
        A numeric offset to add to all span start and end indices. This is helpful
        if the sentence is part of a larger structure, such as a document, which
        the indices need to respect.
    max_span_width : ``int``, optional (default = None)
        The maximum length of spans which should be included. Defaults to len(sentence).
    min_span_width : ``int``, optional (default = 1)
        The minimum length of spans which should be included. Defaults to 1.
    filter_function : ``Callable[[List[T]], bool]``, optional (default = None)
        A function mapping sequences of the passed type T to a boolean value.
        If ``True``, the span is included in the returned spans from the
        sentence, otherwise it is excluded..
    """
    max_span_width = max_span_width or len(sentence)
    filter_function = filter_function or (lambda x: True)
    spans: List[Tuple[int, int]] = []

    for start_index in range(len(sentence)):
        last_end_index = min(start_index + max_span_width, len(sentence))
        first_end_index = min(start_index + min_span_width - 1, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index
            # add 1 to end index because span indices are inclusive.
            if filter_function(sentence[slice(start_index, end_index + 1)]):
                spans.append((start, end))
    return spans
