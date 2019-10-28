from typing import Dict, Optional, List, Tuple
import numpy as np
import pickle
import pdb
from collections import namedtuple, defaultdict

from util.utils import Alphabet


GLOVE_FILE = "./embeddings/glove.6B.100d.txt"
PUBMED_WORD2VEC_FILE = "./embeddings/PubMed-shuffle-win-2.bin"

SentInst = namedtuple('SentInst', 'tokens chars entities')

PREDIFINE_TOKEN_IDS = {'DEFAULT': 0}
PREDIFINE_CHAR_IDS = {'DEFAULT': 0, 'BOT': 1, 'EOT': 2}


class Reader:
    def __init__(self) -> None:

        self.vocab2id: Dict[str, int] = {}
        self.lowercase: Optional[bool] = None

        self.word_iv_alphabet: Optional[Alphabet] = None
        self.word_ooev_alphabet: Optional[Alphabet] = None
        self.char_alphabet: Optional[Alphabet] = None
        self.label_alphabet: Optional[Alphabet] = None

        self.train: Optional[List[SentInst]] = None
        self.dev: Optional[List[SentInst]] = None
        self.test: Optional[List[SentInst]] = None

    def read_and_gen_vectors_glove(self, embed_path: str) -> None:
        token_embed = None
        ret_mat = []
        with open(GLOVE_FILE, 'r') as f:
            id = 0
            for line in f:
                s_s = line.split()
                if token_embed is None:
                    token_embed = len(s_s) - 1
                    ret_mat.append(np.zeros(token_embed).astype('float32'))
                else:
                    assert (token_embed + 1 == len(s_s))
                id += 1
                self.vocab2id[s_s[0]] = id
                ret_mat.append(np.array([float(x) for x in s_s[1:]]))

        self.lowercase = True

        ret_mat = np.array(ret_mat)
        with open(embed_path, 'wb') as f:
            pickle.dump(ret_mat, f)

    def read_and_gen_vectors_pubmed_word2vec(self, embed_path: str) -> None:
        ret_mat = []
        with open(PUBMED_WORD2VEC_FILE, 'rb') as f:
            line = f.readline().rstrip(b'\n')
            vsize, token_embed = line.split()
            vsize = int(vsize)
            token_embed = int(token_embed)
            id = 0
            ret_mat.append(np.zeros(token_embed).astype('float32'))

            for v in range(vsize):
                wchars = []
                while True:
                    c = f.read(1)
                    if c == b' ':
                        break
                    assert (c is not None)
                    wchars.append(c)
                word = b''.join(wchars)
                if word.startswith(b'\n'):
                    word = word[1:]
                id += 1
                self.vocab2id[word.decode('utf-8')] = id
                ret_mat.append(np.fromfile(f, np.float32, token_embed))
            assert (vsize + 1 == len(ret_mat))

        self.lowercase = False

        ret_mat = np.array(ret_mat)
        with open(embed_path, 'wb') as f:
            pickle.dump(ret_mat, f)

    @staticmethod
    def read_file(filename: str, mode: str = 'train') -> List[SentInst]:
        sent_list = []
        max_len = 0
        num_thresh = 0
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line == "":  # last few blank lines
                    break

                raw_tokens = line.split(' ')
                tokens = raw_tokens
                chars = [list(t) for t in raw_tokens]

                entities = next(f).strip()
                if entities == "":  # no entities
                    sent_inst = SentInst(tokens, chars, [])
                else:
                    entity_list = []
                    entities = entities.split("|")
                    for item in entities:
                        pointers, label = item.split()
                        pointers = pointers.split(",")
                        if int(pointers[1]) > len(tokens):
                            pdb.set_trace()
                        span_len = int(pointers[1]) - int(pointers[0])
                        assert (span_len > 0)
                        if span_len > max_len:
                            max_len = span_len
                        if span_len > 6:
                            num_thresh += 1

                        new_entity = (int(pointers[0]), int(pointers[1]), label)
                        # may be duplicate entities in some datasets
                        if (mode == 'train' and new_entity not in entity_list) or (mode != 'train'):
                            entity_list.append(new_entity)

                    # assert len(entity_list) == len(set(entity_list)) # check duplicate
                    sent_inst = SentInst(tokens, chars, entity_list)
                assert next(f).strip() == ""  # separating line

                sent_list.append(sent_inst)
        print("Max length: {}".format(max_len))
        print("Threshold 6: {}".format(num_thresh))
        return sent_list

    def gen_dic(self) -> None:
        word_set = set()
        char_set = set()
        label_set = set()

        for sent_list in [self.train, self.dev, self.test]:
            num_mention = 0
            for sentInst in sent_list:
                if sent_list is self.train:
                    for token in sentInst.chars:
                        for char in token:
                            char_set.add(char)
                    for token in sentInst.tokens:
                        if self.lowercase:
                            token = token.lower()
                        if token not in self.vocab2id:
                            word_set.add(token)
                for entity in sentInst.entities:
                    label_set.add(entity[2])
                num_mention += len(sentInst.entities)
            print("# mentions: {}".format(num_mention))

        self.word_iv_alphabet = Alphabet(self.vocab2id, len(PREDIFINE_TOKEN_IDS))
        self.word_ooev_alphabet = Alphabet(word_set, len(PREDIFINE_TOKEN_IDS))
        self.char_alphabet = Alphabet(char_set, len(PREDIFINE_CHAR_IDS))
        self.label_alphabet = Alphabet(label_set, 0)

    @staticmethod
    def pad_batches(token_iv_batches: List[List[List[int]]],
                    token_ooev_batches: List[List[List[int]]],
                    char_batches: List[List[List[List[int]]]]) \
            -> Tuple[List[List[List[int]]],
                     List[List[List[int]]],
                     List[List[List[List[int]]]],
                     List[List[List[bool]]]]:

        default_token_id = PREDIFINE_TOKEN_IDS['DEFAULT']
        default_char_id = PREDIFINE_CHAR_IDS['DEFAULT']
        bot_id = PREDIFINE_CHAR_IDS['BOT']  # beginning of token
        eot_id = PREDIFINE_CHAR_IDS['EOT']  # end of token

        padded_token_iv_batches = []
        padded_token_ooev_batches = []
        padded_char_batches = []
        mask_batches = []

        all_batches = list(zip(token_iv_batches, token_ooev_batches, char_batches))
        for token_iv_batch, token_ooev_batch, char_batch in all_batches:

            batch_len = len(token_iv_batch)
            max_sent_len = len(token_iv_batch[0])
            max_char_len = max([max([len(t) for t in char_mat]) for char_mat in char_batch])

            padded_token_iv_batch = []
            padded_token_ooev_batch = []
            padded_char_batch = []
            mask_batch = []

            for i in range(batch_len):

                sent_len = len(token_iv_batch[i])

                padded_token_iv_vec = token_iv_batch[i].copy()
                padded_token_iv_vec.extend([default_token_id] * (max_sent_len - sent_len))
                padded_token_ooev_vec = token_ooev_batch[i].copy()
                padded_token_ooev_vec.extend([default_token_id] * (max_sent_len - sent_len))
                padded_char_mat = []
                for t in char_batch[i]:
                    padded_t = list()
                    padded_t.append(bot_id)
                    padded_t.extend(t)
                    padded_t.append(eot_id)
                    padded_t.extend([default_char_id] * (max_char_len - len(t)))
                    padded_char_mat.append(padded_t)
                for t in range(sent_len, max_sent_len):
                    padded_char_mat.append([default_char_id] * (max_char_len + 2))  # max_len + bot + eot
                mask = [True] * sent_len + [False] * (max_sent_len - sent_len)

                padded_token_iv_batch.append(padded_token_iv_vec)
                padded_token_ooev_batch.append(padded_token_ooev_vec)
                padded_char_batch.append(padded_char_mat)
                mask_batch.append(mask)

            padded_token_iv_batches.append(padded_token_iv_batch)
            padded_token_ooev_batches.append(padded_token_ooev_batch)
            padded_char_batches.append(padded_char_batch)
            mask_batches.append(mask_batch)

        return padded_token_iv_batches, padded_token_ooev_batches, padded_char_batches, mask_batches

    def to_batch(self, batch_size: int) -> Tuple:
        ret_list = []

        for sent_list in [self.train, self.dev, self.test]:
            token_iv_dic = defaultdict(list)
            token_ooev_dic = defaultdict(list)
            char_dic = defaultdict(list)
            label_dic = defaultdict(list)

            this_token_iv_batches = []
            this_token_ooev_batches = []
            this_char_batches = []
            this_label_batches = []

            for sentInst in sent_list:

                token_iv_vec = []
                token_ooev_vec = []
                for t in sentInst.tokens:
                    if self.lowercase:
                        t = t.lower()
                    if t in self.vocab2id:
                        token_iv_vec.append(self.vocab2id[t])
                        token_ooev_vec.append(0)
                    else:
                        token_iv_vec.append(0)
                        token_ooev_vec.append(self.word_ooev_alphabet.get_index(t))

                char_mat = [[self.char_alphabet.get_index(c) for c in t] for t in sentInst.chars]
                # max_len = max([len(t) for t in sentInst.chars])
                # char_mat = [ t + [0] * (max_len - len(t)) for t in char_mat ]

                label_list = [(u[0], u[1], self.label_alphabet.get_index(u[2])) for u in sentInst.entities]

                token_iv_dic[len(sentInst.tokens)].append(token_iv_vec)
                token_ooev_dic[len(sentInst.tokens)].append(token_ooev_vec)
                char_dic[len(sentInst.tokens)].append(char_mat)
                label_dic[len(sentInst.tokens)].append(label_list)

            token_iv_batches = []
            token_ooev_batches = []
            char_batches = []
            label_batches = []
            for length in sorted(token_iv_dic.keys(), reverse=True):
                token_iv_batches.extend(token_iv_dic[length])
                token_ooev_batches.extend(token_ooev_dic[length])
                char_batches.extend(char_dic[length])
                label_batches.extend(label_dic[length])

            [this_token_iv_batches.append(token_iv_batches[i:i + batch_size])
             for i in range(0, len(token_iv_batches), batch_size)]
            [this_token_ooev_batches.append(token_ooev_batches[i:i + batch_size])
             for i in range(0, len(token_ooev_batches), batch_size)]
            [this_char_batches.append(char_batches[i:i + batch_size])
             for i in range(0, len(char_batches), batch_size)]
            [this_label_batches.append(label_batches[i:i + batch_size])
             for i in range(0, len(label_batches), batch_size)]

            this_token_iv_batches, this_token_ooev_batches, this_char_batches, this_mask_batches \
                = self.pad_batches(this_token_iv_batches, this_token_ooev_batches, this_char_batches)

            ret_list.append((this_token_iv_batches,
                             this_token_ooev_batches,
                             this_char_batches,
                             this_label_batches,
                             this_mask_batches))

        return tuple(ret_list)

    def read_all_data(self, file_path: str, train_file: str, dev_file: str, test_file: str) -> None:
        self.train = self.read_file(file_path + train_file)
        self.dev = self.read_file(file_path + dev_file, mode='dev')
        self.test = self.read_file(file_path + test_file, mode='test')
        self.gen_dic()

    def debug_single_sample(self,
                            token_v: List[int],
                            char_mat: List[List[int]],
                            char_len_vec: List[int],
                            label_list: List[Tuple[int, int, int]]) -> None:
        print(" ".join([self.word_ooev_alphabet.get_instance(t) for t in token_v]))
        for t in char_mat:
            print(" ".join([self.char_alphabet.get_instance(c) for c in t]))
        print(char_len_vec)
        for label in label_list:
            print(label[0], label[1], self.label_alphabet.get_instance(label[2]))
