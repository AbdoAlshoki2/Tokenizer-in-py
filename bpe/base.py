from typing import Union
import os
import regex
import unicodedata

GPT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def get_stats(byte_ids: list, counter: dict = None) -> dict:
    """
    function to count the frequancy of each adjacent pair of letters in a given text.
    support updating existing counter.
    Parameters
    ----------
    text : str
        input text
    counter : dict, optional
        existing counter, by default None

    Returns
    -------
    dict
        counter
    """
    # Initialize it properly
    if counter is None:
        counter = {}

    for pair in zip(byte_ids, byte_ids[1:]):
        counter[pair] = counter.get(pair, 0) + 1

    return counter


def merge(byte_ids: list , pair: tuple , new_token: int) -> list:
    """
    function to merge a given pair of integers in a given list.
    Parameters
    ----------
    byte_ids : list
        input list
    pair : tuple
        pair of integers
    new_token : int
        new token to replace the pair

    Returns
    -------
    list
        output list, new list after replacement
    """
    new_list =[]
    i = 0
    while i < len(byte_ids) - 1:
        p = (byte_ids[i], byte_ids[i+1])
        if p == pair:
            new_list.append(new_token)
            i += 2
        else:
            new_list.append(byte_ids[i])
            i += 1

    if i < len(byte_ids):
        new_list.append(byte_ids[-1])
    return new_list


def bpe(text : Union[str, list[str]],
        vocab_size: int = 256,
        merges: dict = None):
    """
    function to apply the BPE algorithm to a given text or list of texts.
    Parameters
    ----------
    text : Union[str, list[str]]
        input text or list of texts
    vocab_size : int, optional
        size of the vocabulary, by default 256 (no bpe)
    merges : dict, optional
        existing merges, by default None

    Returns
    -------
    vocab : dict
        vocabulary in form of (token : bytes)
    merges : dict
        merges in form of (pair: new_token)
    """
    assert(vocab_size >= 256)
    merges = {} if merges is None else merges
    ids = []
    text = [text] if isinstance(text,str) else text
    counts = {}
    for t in text:
        t_bytes = t.encode("utf-8")
        ids.append(list(t_bytes))
        counts = get_stats(list(t_bytes),counts)

    i = 0
    while i < vocab_size - 256:
        pair = max(counts, key=counts.get)

        if counts[pair] < 2:
            break
        counts = {}

        if pair in merges:
            for k in range(len(ids)):
                ids[k] = merge(ids[k], pair, merges[pair])
                counts = get_stats(ids[k],counts)
            continue


        new_token = len(merges) + 256
        merges[pair] = new_token

        for k in range(len(ids)):
            ids[k] = merge(ids[k], pair, merges[pair])
            counts = get_stats(ids[k],counts)
        i += 1


    return  merges


def build_vocab(merges: dict):
    """
    function to build the vocabulary from the merges.
    Parameters
    ----------
    merges : dict
        merges in form of (pair: new_token)

    Returns
    -------
    vocab : dict
        vocabulary in form of (token : bytes)
    """
    vocab = {idx : bytes([idx]) for idx in range(256)}
    for (p1,p2), token in merges.items():
        vocab[token] = vocab[p1] + vocab[p2]
    return vocab


    
def encode(text: Union[str, list[str]], merges: dict) -> list:
    """
    function to encode a given text or list of texts to known tokens.
    Parameters:
    -----------
    text : Union[str, list[str]]
        the text or list of texts to be encoded
    merges : dict
        merged tokens map
    
    Returns:
    --------
    ids : list
        the encoded text byte values
    """
    text = [text] if isinstance(text, str) else text
    ids = []
    for t in text:

        byte_ids = list(t.encode("utf-8"))
        

        while len(byte_ids) >= 2:
            counts = get_stats(byte_ids, None)  
            pair = min(counts, key=lambda k: merges.get(k, float("inf")))

            if pair not in merges:
                break
            
            byte_ids = merge(byte_ids, pair, merges[pair])

        ids.append(byte_ids)

    return ids

def decode(ids: Union[list[int], list[list[int]]], vocab: dict) ->list:
    """
    function to decode a given list of ids or list of list of ids to text.
    Parameters:
    -----------
    ids: Union[list[int], list[list[int]]]
        the token values list to be decoded
    vocab: dict
        the vocabulary mapping (token , bytes)

    Returns:
    --------
    text: list
        the decoded text in utf-8
    """
    text = []
    if isinstance(ids[0], list):
        for i in ids:
            
            decoded = b''.join([vocab[idx] for idx in i]).decode("utf-8", errors="replace")
            decoded = replace_control_characters(decoded)
            text.append(decoded)
    else:

        decoded = b''.join([vocab[idx] for idx in ids]).decode("utf-8", errors="replace")
        decoded = replace_control_characters(decoded)
        text.append(decoded)

    return text

def read_raw_text(path :str) -> list[str]:
    """
    function to read raw text from a given path.
    Parameters:
    -----------
    path: str
        the path to a file or directory
    
    Returns:
    --------
    texts: list
        the file(s) contents in utf-8
    """
    assert os.path.exists(path), "path does not exist"
    texts = []
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".txt"): # reads .txt files
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        texts.append(f.read())

    elif os.path.isfile(path) and path.endswith(".txt"): # reads .txt files
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())

    return texts

def read_files(path :str , match_pattern : Union[bool , str]=False) -> list[str]:
    """
    function to read files and allow regex pattern splitting.
    this function uses GPT-4 pattern matching.
    Parameters:
    -----------
    path: str
        the path to a file or directory

    match_pattern : Union[bool , str]
        the regex pattern (if type is str), or to use the defined pattern or not (if)
    
    Returns:
    --------
    texts: list
        the file(s) contents in utf-8
    """
    texts = read_raw_text(path)
    new_texts = []
    if isinstance(match_pattern,bool):
        if match_pattern:
            compiled = regex.compile(GPT_PATTERN)
            for text in texts:
                new_texts.append(compiled.findall(text))
        else:
            return texts
    else:
        compiled = regex.compile(match_pattern)
        for text in texts:
            new_texts.append(compiled.findall(text))

    return new_texts

def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s