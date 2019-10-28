import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as f
from torch.nn.parameter import Parameter


class VarConv1d(nn.Module):
    r"""Applies a 1D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both sides
      for :attr:`padding` number of points.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters,
          of size
          :math:`\left\lfloor\frac{C_\text{out}}{C_\text{in}}\right\rfloor`

    .. note::

        Depending of the size of your kernel, several (of the last)
        columns of the input might be lost, because it is a valid
        `cross-correlation`_, and not a full `cross-correlation`_.
        It is up to the user to add proper padding.

    .. note::

        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.

        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
        :math:`(C_\text{in}=C_{in}, C_\text{out}=C_{in} \times K, ..., \text{groups}=C_{in})`.

    .. include:: cudnn_deterministic.rst

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, Length, C_{in}, L_{in})`
        - Output: :math:`(N, Length, C_{out}, L_{out})` where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            (out_channels, in_channels, kernel_size). The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{1}{C_\text{in} * \text{kernel\_size}}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{1}{C_\text{in} * \text{kernel\_size}}`
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, dropout: float = 0.):
        super(VarConv1d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.padding: int = padding
        self.dilation: int = dilation
        self.groups: int = groups
        self.weight: Tensor = Parameter(Tensor(out_channels, in_channels // groups, kernel_size), True)
        if bias:
            self.bias: Tensor = Parameter(Tensor(out_channels), True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        if dropout < 0. or dropout > 1.:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(dropout))
        self.p: float = dropout
        self.dropout: nn.Module = nn.Dropout(p=self.p)

    def extra_repr(self) -> str:
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.output_padding != 0:
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        for weight in self.parameters():
            if weight.dim() == 1:
                nn.init.constant_(weight, 0.)
            else:
                nn.init.xavier_uniform_(weight)

    def forward(self, input: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            input_size = input.size()
            input = input.view((input_size[0] * input_size[1], input_size[2], input_size[3]))
            output = f.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            output_size = output.size()
            return output.view((input_size[0], input_size[1], output_size[1], output_size[2]))

        output = []
        for i in range(input.size(0)):
            weight = self.dropout(self.weight)
            output.append(f.conv1d(input[i], weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                          .unsqueeze(0))
        return torch.cat(tuple(output), dim=0)
