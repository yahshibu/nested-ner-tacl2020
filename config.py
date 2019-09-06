from __future__ import division

from training.utils import Optimizer


class Config:
    def __init__(self) -> None:
        self.root_path: str = "."

        # for data loader
        self.data_set: str = "sample"
        self.lowercase: bool = None  # TODO: to be deleted
        self.batch_size: int = 32
        self.if_shuffle: bool = True

        # override when loading data
        self.voc_iv_size: int = None
        self.voc_ooev_size: int = None
        self.char_size: int = None
        self.label_size: int = None

        # embed size
        self.token_embed: int = None
        self.char_embed: int = 25
        self.word_dropout: float = 0.05
        self.char_dropout: float = 0.00

        # for cnn
        self.num_filters: int = 50
        self.kernel_size: int = 3
        self.cnn_dropout: float = 0.20

        # for lstm
        self.hidden_size: int = 200
        self.layers: int = 2
        self.lstm_dropout: float = 0.20

        # for training
        self.embed_path: str = self.root_path + "/data/word_vec_{}.pkl".format(self.data_set)
        self.epoch: int = 500
        self.if_gpu: bool = False
        self.opt: Optimizer = Optimizer.AdaBound
        self.lr: float = 0.001 if self.opt != Optimizer.SGD else 0.1
        self.final_lr: float = 0.1 if self.opt == Optimizer.AdaBound else None
        self.l2: float = 0.
        self.check_every: int = 1
        self.clip_norm: int = 5

        # for early stop
        self.lr_patience: int = 5 if self.opt != Optimizer.SGD else 3

        self.data_path: str = self.root_path + "/data/{}".format(self.data_set)
        self.train_data_path: str = self.data_path + "_train.pkl"
        self.dev_data_path: str = self.data_path + "_dev.pkl"
        self.test_data_path: str = self.data_path + "_test.pkl"
        self.config_data_path: str = self.data_path + "_config.pkl"
        self.model_root_path: str = self.root_path + "/dumps"
        self.model_path: str = self.model_root_path + "/{}_model".format(self.data_set)

    def __repr__(self) -> str:
        return str(vars(self))


config: Config = Config()
