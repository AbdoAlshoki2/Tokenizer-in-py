from .base import bpe, build_vocab , encode, decode , read_files, render_token
from typing import Union

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self.pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def train(self , text: str , vocab_size: int , ispath = False , match_pattern= False):

        if ispath:
            text = read_files(text , match_pattern=match_pattern)

        self.merges = bpe(text , vocab_size, self.merges)
        self.vocab = build_vocab(self.merges)
        
    def encode(self, text: Union[str, list[str]]):
        return encode(text , self.merges)
    
    def decode(self, ids: Union[list[int], list[list[int]]]):
        return decode(ids,self.vocab)


    def save(self,file_name: str):
        model_file = file_name + '.model'

        with open(model_file , 'w') as f:
            f.write('bpe v1\n')
            f.write(f'{self.pattern}\n')
            f.write(f'{len(self.merges)}\n')
            for idx1, idx2 in self.merges:
                f.write(f'{idx1} {idx2}\n')

        vocab_file = file_name + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():

                s = render_token(token)

                if idx in inverted_merges:

                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:

                    f.write(f"[{s}] {idx}\n")




    def load(self , path: str):
        assert path.endswith('.model')
        merges = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:

            version = f.readline().strip()
            assert version == "minbpe v1"

            self.pattern = f.readline().strip()

            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        
        self.merges = merges
        self.vocab = build_vocab(self.merges)







