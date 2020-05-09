from typing import List, Dict, Set
from itertools import chain

### You can import any Python standard libraries here.
### Do not import external library such as numpy, torchtext, etc.
from collections import Counter, defaultdict, OrderedDict
import re
### END YOUR LIBRARIES

def build_bpe(
    corpus: List[str],
    max_vocab_size: int
) -> List[int]:
    """ BPE Vocabulary Builder
    Implement vocabulary builder for byte pair encoding.
    Please sort your idx2word by subword length in decsending manner.

    Hint: Counter in collection library would be helpful

    Note: If you convert sentences list to word frequence dictionary,
          building speed is enhanced significantly because duplicated words are preprocessed together

    Arguments:
    corpus -- List of words to build vocab
    max_vocab_size -- The maximum size of vocab

    Return:
    idx2word -- Subword list
    """
    # Special tokens
    PAD = BytePairEncoding.PAD_token # Index of <PAD> must be 0
    UNK = BytePairEncoding.UNK_token # Index of <UNK> must be 1
    CLS = BytePairEncoding.CLS_token # Index of <CLS> must be 2
    SEP = BytePairEncoding.SEP_token # Index of <SEP> must be 3
    MSK = BytePairEncoding.MSK_token # Index of <MSK> must be 4
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]

    WORD_END = BytePairEncoding.WORD_END # Use this token as the end of a word

    ### YOUR CODE HERE (~22 lines)
    idx2word: List[str] = SPECIAL
    if type(max_vocab_size) is not int: #  Fix max_vocab_size coming in as dtype != int
        max_vocab_size = int(max_vocab_size)
    corpus = [c + WORD_END for c in corpus] # Append WORD_END to every other word
    corpus = [" ".join(s) for s in corpus] # Add space between each character for each word
    vocab = list(OrderedDict.fromkeys(chain.from_iterable(corpus))) # Extract unique characters for initial vocab
    vocab.remove(' ')
    word2cnt = dict(Counter(corpus))
    
    while len(vocab) + len(SPECIAL) < max_vocab_size:
        pairs = defaultdict(int)
        for word, freq in word2cnt.items(): # get stats of each subword pair (frequency)
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq # Record co-occurring pair of unigrams
        if len(pairs) >= 1:
            largest = max(pairs, key=lambda k: pairs[k])
        bigram = ' '.join(largest) # Merge the pairs based on the merge rule (Rule: Merge the maximum occurrences first!)
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') # Specify patterns to strip off the bigram
        v_out = {}
        prev_vocab = vocab.copy()
        for word in word2cnt:
            joined_bigram = ''.join(largest)
            w_new = p.sub(joined_bigram, word) # Replace each pattern in "word" given by 'p' with the chosen bigram
            vocab.append(joined_bigram) if joined_bigram not in vocab else vocab
            v_out[w_new] = word2cnt[word]
        if prev_vocab == vocab: # Check if vocab did not change from the previous iteration
            break
        word2cnt = v_out.copy()
    idx2word.extend(vocab)
    idx2word = idx2word[:5] + sorted(idx2word[5:], key=len, reverse=True)
    return idx2word
    ### END YOUR CODE

    return idx2word

def encode(
    sentence: List[str],
    idx2word: List[str]
) -> List[int]:
    """ BPE encoder
    Implement byte pair encoder which takes a sentence and gives the encoded tokens

    Arguments:
    sentence -- The list of words which need to be encoded.
    idx2word -- The vocab that you have made on the above build_bpe function.
    
    Return:
    tokens -- The list of the encoded tokens
    """
    WORD_END = BytePairEncoding.WORD_END

    ### YOUR CODE HERE (~10 lines)
    tokens: List[int] = None
    if len(sentence) < 1 or len(idx2word) < 1:
        return []
    tokens = []
    for word in sentence:
        for tok in range(len(idx2word)):
            try:
                idx = word.index(idx2word[tok])
            except ValueError:
                continue
            else:
                word = word.replace(idx2word[tok], str(tok) + ' ')
        tokens += list(map(int, word.split())) + [idx2word.index(WORD_END)]
    ### END YOUR CODE

    return tokens

def decode(
    tokens: List[int],
    idx2word: List[str]
) -> List[str]:
    """ BPE decoder
    Implement byte pair decoder which takes tokens and gives the decoded sentence.

    Arguments:
    tokens -- The list of tokens which need to be decoded
    idx2word -- the vocab that you have made on the above build_bpe function.

    Return:
    sentence  -- The list of the decoded words
    """
    WORD_END = BytePairEncoding.WORD_END

    ### YOUR CODE HERE (~1 lines)
    sentence: List[str] = None
    sentence = list(filter(lambda x: x != '', ''.join([idx2word[tok] for tok in tokens]).split(WORD_END)))
    ### END YOUR CODE
    return sentence


#############################################
# Helper functions below. DO NOT MODIFY!    #
#############################################

class BytePairEncoding(object):
    """ Byte Pair Encoding class
    We aren't gonna use this class for encoding. Because it is too slow......
    We will use sentence piece Google have made.
    Thus, this class is just for special token index reference.
    """
    PAD_token = '<pad>'
    PAD_token_idx = 0
    UNK_token = '<unk>'
    UNK_token_idx = 1
    CLS_token = '<cls>'
    CLS_token_idx = 2
    SEP_token = '<sep>'
    SEP_token_idx = 3
    MSK_token = '<msk>'
    MSK_token_idx = 4

    WORD_END = '_'

    def __init__(self, corpus: List[List[str]], max_vocab_size: int) -> None:
        self.idx2word = build_bpe(corpus, max_vocab_size)

    def encode(self, sentence: List[str]) -> List[int]:
        return encode(sentence, self.idx2word)

    def decoder(self, tokens: List[int]) -> List[str]:
        return decode(tokens, self.idx2word)
    
#############################################
# Testing functions below.                  #
#############################################

def test_build_bpe():
    print ("======Building BPE Vocab Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    WORD_END = BytePairEncoding.WORD_END

    # First test
    corpus = ['abcde']
    vocab = build_bpe(corpus, max_vocab_size=15)
    assert vocab[:5] == [PAD, UNK, CLS, SEP, MSK], \
        "Please insert the special tokens properly"
    print("The first test passed!")

    # Second test
    assert sorted(vocab[5:], key=len, reverse=True) == vocab[5:], \
        "Please sort your idx2word by subword length in decsending manner."
    print("The second test passed!")

    # Third test
    corpus = ['low'] * 5 + ['lower'] * 2 + ['newest'] * 6 + ['widest'] * 3
    vocab = set(build_bpe(corpus, max_vocab_size=24))
    assert vocab > {PAD, UNK, CLS, SEP, MSK, 'est_', 'low', 'newest_', \
                    'i', 'e', 'n', 't', 'd', 's', 'o', 'l', 'r', 'w', WORD_END} and \
           "low_" not in vocab and "wi" not in vocab and "id" not in vocab, \
           "Your bpe result does not match expected result"
    print("The third test passed!")

    # forth test
    corpus = ['aaaaaaaaaaaa', 'abababab']
    vocab = set(build_bpe(corpus, max_vocab_size=13))
    assert vocab == {PAD, UNK, CLS, SEP, MSK, 'aaaaaaaa', 'aaaa', 'abab', 'aa', 'ab', 'a', 'b', WORD_END}, \
           "Your bpe result does not match expected result"
    print("The forth test passed!")

    # fifth test
    corpus = ['abc', 'bcd']
    vocab = build_bpe(corpus, max_vocab_size=10000)
    assert len(vocab) == 15, \
           "Your bpe result does not match expected result"
    print("The fifth test passed!")

    print("All 5 tests passed!")

def test_encoding():
    print ("======Encoding Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]
    WORD_END = BytePairEncoding.WORD_END

    # First test
    vocab = SPECIAL + ['bcc', 'bb', 'bc', 'a', 'b', 'c', WORD_END]
    assert encode(['abbccc'], vocab) == [8, 9, 5, 10, 11], \
           "Your bpe encoding does not math expected result"
    print("The first test passed!")

    # Second test
    vocab = SPECIAL + ['aaaa', 'aa', 'a', WORD_END]
    assert len(encode(['aaaaaaaa', 'aaaaaaa'], vocab)) == 7, \
           "Your bpe encoding does not math expected result"
    print("The second test passed!")

    print("All 2 tests passed!")

def test_decoding():
    print ("======Decoding Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]
    WORD_END = BytePairEncoding.WORD_END

    # First test
    vocab = SPECIAL + ['bcc', 'bb', 'bc', 'a', 'b', 'c', WORD_END]
    assert decode([8, 9, 5, 10, 11], vocab) == ['abbccc'], \
           "Your bpe decoding does not math expected result"
    print("The first test passed!")

    # Second test
    vocab = SPECIAL + ['aaaa', 'aa', 'a', WORD_END]
    assert decode([5, 5, 8, 5, 6, 7, 8], vocab) == ['aaaaaaaa', 'aaaaaaa'], \
           "Your BPE decoding does not math expected result"
    print("The second test passed!")

def test_consistency():
    print ("======Consistency Test Case======")
    corpus = ['this is test corpus .',
              'we will check the consistency of your byte pairing encoding .', 
              'you have to pass this test to get full scores .',
              'we hope you to pass tests wihtout any problem .',
              'good luck .']

    vocab = build_bpe(chain.from_iterable(sentence.split() for sentence in corpus), 80)
    
    sentence = 'this is another sentence to test encoding and decoding .'.split()

    assert decode(encode(sentence, vocab), vocab) == sentence, \
            "Your BPE does not show consistency."
    print("The consistency test passed!")

if __name__ == "__main__":
    test_build_bpe()
    test_encoding()
    test_decoding()
    test_consistency()
    