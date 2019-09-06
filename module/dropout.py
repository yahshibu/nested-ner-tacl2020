import torch.nn as nn
from torch.tensor import Tensor


class WordDropout(nn.Module):
    def __init__(self, p: float = 0.05) -> None:
        super(WordDropout, self).__init__()
        self.p: float = p

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: Tensor
                the input tensor with shape = [batch, length, word_dim]

        Returns: Tensor
            the tensor with shape = [batch, length, word_dim]
        """
        if self.p == 0. or not self.training:
            return input

        batch, length, _ = input.size()
        m = input.new_empty((batch, length, 1)).bernoulli_(1. - self.p)
        return m * input


class CharDropout(nn.Module):
    def __init__(self, p: float = 0.00) -> None:
        super(CharDropout, self).__init__()
        self.p: float = p

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: Tensor
                the input tensor with shape = [batch, length, char_length, char_dim]

        Returns: Tensor
            the tensor with shape = [batch, length, char_length, char_dim]
        """
        if self.p == 0. or not self.training:
            return input

        batch, length, char_length, _ = input.size()
        m = input.new_empty((batch, length, char_length, 1)).bernoulli_(1. - self.p)
        return m * input
