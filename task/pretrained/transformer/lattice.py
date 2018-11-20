# -------------------------------------------------------- lattice -------------------------------------
import torch
from torch import nn
import numpy as np
from collections import namedtuple
from typing import List, Union, Set
import tarfile
from task.util import utils
from tqdm import tqdm
from torchtext import data
from sklearn.utils import murmurhash3_32
from .base import clones, AddNormLayer, PositionwiseFeedForward
import gzip
import math

WordInfo = namedtuple('WordInfo', ['id', 'left', 'right'])


def normalize(matrix):
    norm = np.sqrt(np.sum(matrix * matrix, axis=1))
    matrix = matrix / norm[:, np.newaxis]
    return matrix


def decay_factor(rankid: int):
    if rankid < 100000:
        return 1.0
    elif rankid < 500000:
        return 0.8
    elif rankid < 1000000:
        return 0.6
    elif rankid < 2000000:
        return 0.4
    elif rankid < 4000000:
        return 0.2
    else:
        return 0.1


def read_vectors(path):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    iw = []
    wi = {}
    with gzip.open(path, mode='rt', compresslevel=6) as f:
        head = f.readline()
        word_size, dim = [int(i) for i in head.rstrip().split()]
        matrix = np.zeros(shape=(1000000, dim), dtype=np.float32)
        for i, line in tqdm(enumerate(f)):
            lines_num += 1
            try:
                line_id, word, weights = line.rstrip().split('\t')
                matrix[i, :] = np.asarray([float(x) for x in weights.split()],
                                          dtype=np.float32) * decay_factor(int(line_id))
                iw.append(word)
            except Exception:
                print(line)
                print(line.rstrip().split('\t'))

            if len(iw) >= 1000000:
                break

    for i, w in enumerate(iw):
        wi[w] = i

    # Turn vectors into numpy format and normalize them
    # matrix = normalize(matrix)

    return matrix, iw, wi, dim


def hash_next(key: int, subword: Union[str, int]):
    next = murmurhash3_32(subword) if isinstance(subword, str) else subword
    if key is None:
        key = next
    else:
        key = (key * 8978948897894561157) ^ ((1 + next) * 17894857484156487943);
    return key


def hash(word: str):
    current = None
    for t, subword in utils.replace_entity(word):
        current = hash_next(current, subword)
    return current


class LatticeField(data.Field):
    def __init__(self, embed_path: str):
        super(LatticeField, self).__init__()
        matrix, iw, wi, dim = read_vectors(embed_path)
        self.word_emb = nn.Embedding.from_pretrained(torch.from_numpy(matrix), freeze=True)
        print('load pretrain word embedding finished.')
        self.dim = dim
        self.id2word = iw
        self.word2id = wi
        self.hash2id, self.max_sub_len = self.make_hash(iw)

        print('max_subword_len', self.max_sub_len)

    def make_hash(self, iw):
        max_len = 0

        hash2id = {}
        for i, w in enumerate(iw):
            words = list(utils.replace_entity(w))
            max_len = max(max_len, len(words))
            current = None
            for t, subword in words:
                current = hash_next(current, subword)
            hash2id[current] = i
        return hash2id, max_len

    def process(self, seqs, device=None, train=True):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (-1 or None): Device to create the Variable's Tensor on.
                Use -1 for CPU and None for the currently active GPU device.
                Default: None.
            train (boolean): Whether the batch is for a training set.
                If False, the Variable will be created with volatile=True.
                Default: True.
        """

        begins = []
        middles = []
        ends = []
        for seq in seqs:
            begin, middle, end = [[] for _ in range(len(seq))], [[] for _ in range(len(seq))], [[] for _ in range(len(seq))]
            mm_seq = [murmurhash3_32(s) for s in seq]
            for t in range(len(seq)):
                hash_id = None
                for l in range(min(self.max_sub_len, len(mm_seq) - t)):
                    hash_id = hash_next(hash_id, mm_seq[t + l])
                    word_id = self.hash2id.get(hash_id)
                    if word_id:
                        wordinfo = WordInfo(word_id, t, t+l)
                        begin[t].append(wordinfo)
                        if l > 0:
                            end[t+l].append(wordinfo)
                        if l > 1:
                            for m in range(t+1, t+l, 1):
                                middle[m].append(wordinfo)

            begins.append(begin)
            middles.append(middle)
            ends.append(end)
        begin_lens = [[len(t) for t in b] for b in begins]
        middle_lens = [[len(t) for t in m] for m in middles]
        end_lens = [[len(t) for t in e] for e in ends]

        begin_max_len = max(t for b in begin_lens for t in b)
        middle_max_len = max(t for b in middle_lens for t in b)
        end_max_len = max(t for b in end_lens for t in b)

        batch_size, seq_len = len(seqs), max(len(s) for s in seqs)

        begin_t, begin_left, begin_right = [
            torch.LongTensor(batch_size, seq_len, begin_max_len).zero_() for _ in range(3)] \
            if begin_max_len > 0 else [None, None, None]
        middle_t, middle_left, middle_right = [
            torch.LongTensor(batch_size, seq_len, middle_max_len).zero_() for _ in range(3)] \
            if middle_max_len > 0 else [None, None, None]
        end_t, end_left, end_right = [
            torch.LongTensor(batch_size, seq_len, end_max_len).zero_() for _ in range(3)] \
            if end_max_len > 0 else [None, None, None]

        for b in range(batch_size):
            for t in range(len(seqs[b])):
                if begin_t is not None:
                    for w in range(len(begins[b][t])):
                        begin_t[b, t, w] = begins[b][t][w].id
                        begin_left[b, t, w] = begins[b][t][w].left
                        begin_right[b, t, w] = begins[b][t][w].right
                if middle_t is not None:
                    for w in range(len(middles[b][t])):
                        middle_t[b, t, w] = middles[b][t][w].id
                        middle_left[b, t, w] = middles[b][t][w].left
                        middle_right[b, t, w] = middles[b][t][w].right
                if end_t is not None:
                    for w in range(len(ends[b][t])):
                        end_t[b, t, w] = ends[b][t][w].id
                        end_left[b, t, w] = ends[b][t][w].left
                        end_right[b, t, w] = ends[b][t][w].right

        return (begin_t, begin_left, begin_right, begin_lens), \
               (middle_t, middle_left, middle_right, middle_lens), \
               (end_t, end_left, end_right, end_lens)


class LatticeLayer(nn.Module):
    def __init__(self, hidden_dim, word_emb: nn.Embedding, max_subword_len):
        super(LatticeLayer, self).__init__()
        self.word_emb = word_emb
        self.hidden_dim = hidden_dim
        self.word_size = self.word_emb.num_embeddings
        self.emb_dim = self.word_emb.embedding_dim
        self.max_subword_len = max_subword_len

        self.context_linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.hidden_linear = nn.Linear(self.hidden_dim, self.hidden_dim * 3)

        self.begin_linear = nn.Linear(self.emb_dim, self.hidden_dim * 2)
        self.middle_linear = nn.Linear(self.emb_dim, self.hidden_dim * 2)
        self.end_linear = nn.Linear(self.emb_dim, self.hidden_dim * 2)

    def forward(self, hiddens, masks, lattices):
        """
        :param hiddens: Tensor(len, batch_size, diim)
        :param lattices: ((Tensor(batch_size, seq_len, word_len), List[List[int]]),
                          (Tensor(batch_size, seq_len, word_len), List[List[int]]),
                          (Tensor(batch_size, seq_len, word_len), List[List[int]]))
        :return:
        """
        batch_size, max_len, hidden_dim = hiddens.size()
        assert hidden_dim == self.hidden_dim

        qkvm = [self._make_query_key_value(w, hiddens, l)
               for w, l in zip(lattices, [self.begin_linear, self.middle_linear, self.end_linear])]

        hq, hk, hv = self.hidden_linear(hiddens).split(hidden_dim, -1)

        qkvm = qkvm + [(hq.unsqueeze(-2), hk.unsqueeze(-2), hv.unsqueeze(-2), masks.unsqueeze(-1))]

        querys = torch.cat([q for q, _, _, _ in qkvm if q is not None], dim=-2)
        keys = torch.cat([k for _, k, _, _ in qkvm if k is not None], dim=-2)
        values = torch.cat([v for _, _, v, _ in qkvm if v is not None], dim=-2)
        masks = torch.cat([m for _, _, _, m in qkvm if m is not None], dim=-1)

        weights = (querys * keys).sum(-1).masked_fill_(masks, -1e10).softmax(-1)

        return torch.matmul(weights.unsqueeze(-2), values).squeeze(-2)

    def _make_mask(self, words, lens):
        assert words.dim() == 3
        masks = torch.zeros(words.size(), dtype=torch.uint8)
        for b in range(len(lens)):
            for t in range(len(lens[b])):
                masks[b, t, :lens[b][t]] = 1

        return masks

    def _make_query_key_value(self, lattice, hidden, word_linear):

        words, lefts, rights, lens = lattice

        if words is None:
            return None, None, None, None

        batch_size, seq_len, word_len = words.size()
        _, _, hidden_dim = hidden.size()

        lefts = hidden.gather(1, lefts.view(batch_size, -1, 1).expand(-1, -1, hidden_dim))\
            .view(batch_size, seq_len, word_len, -1)
        rights = hidden.gather(1, rights.view(batch_size, -1, 1).expand(-1, -1, hidden_dim))\
            .view(batch_size, seq_len, word_len, -1)
        querys = self.context_linear(torch.cat([lefts, rights], dim=-1))
        key, value = word_linear(self.word_emb(words)).split(hidden_dim, dim=-1)

        return querys, key, value, self._make_mask(words, lens)


class Gated(nn.Module):
    def __init__(self, dim):
        super(Gated, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(self.dim, 9)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden, masks):

        batch_size, seq_len, _ = hidden.size()
        if seq_len >= 3:
            left, middle, right = self.linear(hidden).split(3, dim=-1)
            begin, middle, end = [t.squeeze(-1)
                                  for t in self.sigmoid(left[:, :-2] + middle[:, 1:-1] + right[:, 2:]).split(1, dim=-1)]

            begin_gated = torch.cat([begin.new_ones(batch_size, 1),
                                     begin,
                                     begin.new_ones(batch_size, 1)], dim=1).masked_fill(masks==0, 0)
            middle_gated = torch.cat([middle.new_zeros(batch_size, 1),
                                      middle,
                                      middle.new_zeros(batch_size, 1)], dim=1).masked_fill(masks==0, 0)
            end_gated = torch.cat([end.new_ones(batch_size, 1),
                                   end,
                                   end.new_ones(batch_size, 1)], dim=1).masked_fill(masks==0, 0)
        else:
            begin_gated = hidden.new_ones(batch_size, seq_len)
            middle_gated = hidden.new_ones(batch_size, seq_len)
            end_gated = hidden.new_ones(batch_size, seq_len)

        return begin_gated, middle_gated, end_gated


class GatedLatticeLayer(nn.Module):
    def __init__(self, hidden_dim, word_emb: nn.Embedding, max_subword_len):
        super(GatedLatticeLayer, self).__init__()
        self.word_emb = word_emb
        self.hidden_dim = hidden_dim
        self.word_size = self.word_emb.num_embeddings
        self.emb_dim = self.word_emb.embedding_dim
        self.max_subword_len = max_subword_len

        self.hidden_linear = nn.Linear(self.hidden_dim, self.hidden_dim * 3)

        self.begin_linear = nn.Linear(self.emb_dim, self.hidden_dim * 2)
        self.middle_linear = nn.Linear(self.emb_dim, self.hidden_dim * 2)
        self.end_linear = nn.Linear(self.emb_dim, self.hidden_dim * 2)

        self.gated_layer = Gated(self.hidden_dim)

    def forward(self, hiddens, masks, lattices):
        """
        :param hiddens: Tensor(len, batch_size, diim)
        :param lattices: ((Tensor(batch_size, seq_len, word_len), List[List[int]]),
                          (Tensor(batch_size, seq_len, word_len), List[List[int]]),
                          (Tensor(batch_size, seq_len, word_len), List[List[int]]))
        :return:
        """
        batch_size, max_len, hidden_dim = hiddens.size()
        assert hidden_dim == self.hidden_dim

        (begins, begin_lens), (middles, middle_lens), (ends, end_lens) = lattices

        begin_masks = self._make_mask(begins, begin_lens) if begins is not None else None
        middle_masks = self._make_mask(middles, middle_lens) if middles is not None else None
        end_masks = self._make_mask(ends, end_lens) if ends is not None else None

        begins = self.begin_linear(self.word_emb(begins)) if begins is not None else None
        middles = self.middle_linear(self.word_emb(middles)) if middles is not None else None
        ends = self.end_linear(self.word_emb(ends)) if ends is not None else None

        query, hidden_key, hidden_value = self.hidden_linear(hiddens).split(hidden_dim, -1)

        begin_gated, middle_gated, end_gated = self.gated_layer(hiddens, masks)

        begins = begin_gated.unsqueeze(-1).unsqueeze(-1) * begins if begins is not None else None

        middles = middle_gated.unsqueeze(-1).unsqueeze(-1) * middles if middles is not None else None

        ends = end_gated.unsqueeze(-1).unsqueeze(-1) * ends if ends is not None else None

        keys, values = torch.cat(
            list(filter(lambda x: x is not None, [begins, middles, ends])), dim=-2).split(hidden_dim, -1)
        keys = torch.cat([hidden_key.unsqueeze(-2), keys], dim=-2)
        values = torch.cat([hidden_value.unsqueeze(-2), values], dim=-2)
        masks = torch.cat(
            list(filter(lambda x: x is not None,
                        [masks.unsqueeze(-1), begin_masks, middle_masks, end_masks])),
            dim=-1)

        scores = torch.matmul(query.unsqueeze(-2), keys.transpose(-1, -2)).squeeze(-2)

        scores.masked_fill_(masks, -1e10)

        weights = scores.softmax(-1)

        return torch.matmul(weights.unsqueeze(-2), values).squeeze(-2)

    def _make_mask(self, words, lens):
        assert words.dim() == 3
        masks = torch.zeros(words.size(), dtype=torch.uint8)
        for b in range(len(lens)):
            for t in range(len(lens[b])):
                masks[b, t, :lens[b][t]] = 1

        return masks


# lattice attention -> (add + norm) -> ffn -> (add + norm)
class LatticeEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, hidden_dim, word_emb: nn.Embedding, max_subword_len, feed_forward, dropout):
        super(LatticeEncoderLayer, self).__init__()
        self.lattice_attn = LatticeLayer(hidden_dim, word_emb, max_subword_len)
        self.feed_forward = feed_forward
        self.sublayer = clones(AddNormLayer(hidden_dim, dropout), 2)
        self.hidden = hidden_dim

    def forward(self, hiddens, masks, lattices):
        "Follow Figure 1 (left) for connections."
        hiddens = self.sublayer[0](hiddens, lambda h: self.lattice_attn(h, masks, lattices))
        return self.sublayer[1](hiddens, self.feed_forward)


if __name__ == '__main__':
    lattice = Lattice('wordvec/Tencent_AILab_ChineseEmbedding.txt')
    latticeLayer = LatticeLayer(lattice.dim, lattice.word_emb, lattice.max_sub_len)

    print(lattice.find([['你', '们', '好']]))


    hiddens = torch.randn(3, 1, lattice.dim)
    print(latticeLayer(hiddens, [3], lattice.find([['你', '们', '好']])))