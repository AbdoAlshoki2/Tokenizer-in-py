from typing import Union
def get_stats(byte_ids: list  , counter: dict = {}) -> dict:
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

    for pair in zip(byte_ids , byte_ids[1:]):
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
        t = t.encode("utf-8")
        ids.append(list(t))
        counts = get_stats(list(t),counts)

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


    
def encode(text: Union[str, list[str]], merges: dict):
    """
    function to encode a given text or list of texts to known tokens.
    """
    text = [text] if isinstance(text,str) else text
    ids = []
    for t in text:
        byte_ids = list(t.encode("utf-8"))
        cnt = 0
        while len(byte_ids) >= 2:
            counts = get_stats(byte_ids , {})
            pair = min(counts , key= lambda k: merges.get(k, float("inf")))

            if pair not in merges:
                break
            
            byte_ids = merge(byte_ids, pair, merges[pair])


        ids.append(byte_ids)

    return ids


def decode(ids: Union[list[int], list[list[int]]] , vocab: dict):
    """
    function to decode a given list of ids or list of list of ids to text.
    """
    text = []
    if isinstance(ids[0],list):
        for i in ids:
            text.append("".join([vocab[idx].decode("utf-8",errors="replace") for idx in i]))
    else:
        text.append("".join([vocab[idx].decode("utf-8",errors="replace") for idx in ids]))

    return text

