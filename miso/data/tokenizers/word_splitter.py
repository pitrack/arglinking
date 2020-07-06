import re
from typing import List

from overrides import overrides
import spacy

from miso.utils.registrable import Registrable
from miso.utils.file import get_spacy_model
from miso.data.tokenizers.token import Token


class WordSplitter(Registrable):
    """
    A ``WordSplitter`` splits strings into words.  This is typically called a "tokenizer" in NLP,
    because splitting strings into characters is trivial, but we use ``Tokenizer`` to refer to the
    higher-level object that splits strings into tokens (which could just be character tokens).
    So, we're using "word splitter" here for this.
    """
    default_implementation = 'spacy'

    def batch_split_words(self, sentences: List[str]) -> List[List[Token]]:
        """
        Spacy needs to do batch processing, or it can be really slow.  This method lets you take
        advantage of that if you want.  Default implementation is to just iterate of the sentences
        and call ``split_words``, but the ``SpacyWordSplitter`` will actually do batched
        processing.
        """
        return [self.split_words(sentence) for sentence in sentences]

    def split_words(self, sentence: str) -> List[Token]:
        """
        Splits ``sentence`` into a list of :class:`Token` objects.
        """
        raise NotImplementedError


@WordSplitter.register('just_spaces')
class JustSpacesWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` that assumes you've already done your own tokenization somehow and have
    separated the tokens by spaces.  We just split the input string on whitespace and return the
    resulting list.  We use a somewhat odd name here to avoid coming too close to the more
    commonly used ``SpacyWordSplitter``.

    Note that we use ``sentence.split()``, which means that the amount of whitespace between the
    tokens does not matter.  This will never result in spaces being included as tokens.
    """
    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        return [Token(t) for t in sentence.split()]


def _remove_spaces(tokens: List[spacy.tokens.Token]) -> List[spacy.tokens.Token]:
    return [token for token in tokens if not token.is_space]

@WordSplitter.register('spacy')
class SpacyWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` that uses spaCy's tokenizer.  It's fast and reasonable - this is the
    recommended ``WordSplitter``.
    """
    def __init__(self,
                 language: str = 'en_core_web_sm',
                 pos_tags: bool = False,
                 parse: bool = False,
                 ner: bool = False) -> None:
        self.spacy = get_spacy_model(language, pos_tags, parse, ner)

    @overrides
    def batch_split_words(self, sentences: List[str]) -> List[List[Token]]:
        return [_remove_spaces(tokens)
                for tokens in self.spacy.pipe(sentences, n_threads=-1)]

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        # This works because our Token class matches spacy's.
        return _remove_spaces(self.spacy(sentence))
