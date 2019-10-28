__author__ = 'max'
__maintainer__ = 'takashi'

from typing import List, Union
import torch
from torch import Tensor
import torch.nn as nn

from module.crf import ChainCRF4NestedNER
from module.dropout import WordDropout, CharDropout
from module.variational_cnn import VarConv1d
from module.variational_rnn import VarMaskedFastLSTM


class NestedSequenceLabel:
    def __init__(self, start: int, end: int, label: Tensor, children: List) -> None:
        self.start = start
        self.end = end
        self.label = label
        self.children = children


class BiRecurrentConvCRF4NestedNER(nn.Module):
    def __init__(self, token_embed: int, voc_iv_size: int, voc_ooev_size: int, char_embed: int, char_size: int,
                 num_filters: int, kernel_size: int, label_size: int, embedd_word: Tensor, hidden_size: int = 200,
                 layers: int = 2, word_dropout: float = 0.20, char_dropout: float = 0.00,
                 cnn_dropout: float = 0.20, lstm_dropout: float = 0.20) -> None:
        super(BiRecurrentConvCRF4NestedNER, self).__init__()

        self.word_embedd_iv: nn.Embedding = nn.Embedding(voc_iv_size, token_embed, _weight=embedd_word)
        self.word_embedd_iv.weight.requires_grad = False
        self.word_embedd_ooev: nn.Embedding = nn.Embedding(voc_ooev_size, token_embed)
        self.char_embedd: nn.Embedding = nn.Embedding(char_size, char_embed)
        self.conv1d: VarConv1d = VarConv1d(char_embed, num_filters, kernel_size,
                                           padding=kernel_size - 1, dropout=cnn_dropout)
        # dropout word
        self.dropout_word: WordDropout = WordDropout(p=word_dropout)
        self.dropout_char: CharDropout = CharDropout(p=char_dropout)
        # standard dropout
        self.dropout_out: nn.Dropout2d = nn.Dropout2d(p=lstm_dropout)

        self.rnn: VarMaskedFastLSTM = VarMaskedFastLSTM(token_embed + num_filters, hidden_size, num_layers=layers,
                                                        batch_first=True, bidirectional=True,
                                                        dropout=(lstm_dropout, lstm_dropout))

        self.reset_parameters()

        self.all_crfs: List[ChainCRF4NestedNER] = []

        for label in range(label_size):
            crf = ChainCRF4NestedNER(hidden_size * 2, 1)
            self.all_crfs.append(crf)
            self.add_module('crf%d' % label, crf)

        self.b_id: int = 0
        self.i_id: int = 1
        self.e_id: int = 2
        self.s_id: int = 3
        self.o_id: int = 4
        self.eos_id: int = 5

    def reset_parameters(self) -> None:
        nn.init.constant_(self.word_embedd_ooev.weight, 0.)
        nn.init.constant_(self.char_embedd.weight, 0.)
        nn.init.xavier_uniform_(self.conv1d.weight)
        nn.init.constant_(self.conv1d.bias, 0.)
        for name, parameter in self.rnn.named_parameters():
            nn.init.constant_(parameter, 0.)
            if name.find('weight_ih') > 0:
                if name.startswith('cell0.weight_ih') or name.startswith('cell1.weight_ih'):
                    bound = (6. / (self.rnn.input_size + self.rnn.hidden_size)) ** 0.5
                else:
                    bound = (6. / ((2 * self.rnn.hidden_size) + self.rnn.hidden_size)) ** 0.5
                nn.init.uniform_(parameter, -bound, bound)
                parameter.data[:2, :, :] = 0.
                parameter.data[3:, :, :] = 0.
            if name.find('bias_hh') > 0:
                parameter.data[1, :] = 1.

    def _get_rnn_output(self, input_word_iv: Tensor, input_word_ooev: Tensor, input_char: Tensor, mask: Tensor = None) \
            -> Tensor:
        # [batch, length, word_dim]
        word = self.word_embedd_iv(input_word_iv) \
               + (input_word_ooev != 0).float().unsqueeze(2) * self.word_embedd_ooev(input_word_ooev)
        word = self.dropout_word(word)

        # [batch, length, char_length, char_dim]
        char = (input_char != 0).float().unsqueeze(3) * self.char_embedd(input_char)
        char = self.dropout_char(char)
        # transpose to [batch, length, char_dim, char_length]
        char = char.transpose(2, 3)
        # put into cnn [batch, length, char_filters, char_length]
        # then put into maxpooling [batch, length, char_filters]
        char, _ = self.conv1d(char).max(dim=3)
        #
        char = torch.sigmoid(char)

        # concatenate word and char [batch, length, word_dim+char_filter]
        input = torch.cat((word, char), dim=2)
        # output from rnn [batch, length, hidden_size]
        output, hn = self.rnn(input, mask)

        # apply dropout for the output of rnn
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output

    def forward(self, input_word_iv: Tensor, input_word_ooev: Tensor, input_char: Tensor,
                target: Union[List[List[NestedSequenceLabel]], List[NestedSequenceLabel]], mask: Tensor) -> Tensor:
        # output from rnn [batch, length, tag_space]
        output = self._get_rnn_output(input_word_iv, input_word_ooev, input_char, mask=mask)

        # [batch, length, num_label, num_label]
        batch, length, _ = output.size()

        loss = []

        for label in range(len(self.all_crfs)):
            target_batch = output.new_empty((batch, length)).long()
            for i in range(batch):
                target_batch[i, :] = target[label][i].label

            loss_batch, energy_batch = self.all_crfs[label].loss(output, target_batch, mask=mask)

            calc_nests_loss = self.all_crfs[label].nests_loss

            def forward_recursively(loss: Tensor, energy: Tensor, target: NestedSequenceLabel, offset: int):
                nests_loss_list = []
                for child in target.children:
                    if child.end - child.start > 1:
                        nests_loss = calc_nests_loss(energy[child.start - offset:child.end - offset, :, :],
                                                     child.label)
                        nests_loss_list.append(forward_recursively(nests_loss,
                                                                   energy[child.start - offset:child.end - offset, :, :],
                                                                   child, child.start))
                return sum(nests_loss_list) + loss

            loss_each = []
            for i in range(batch):
                loss_each.append(forward_recursively(loss_batch[i], energy_batch[i], target[label][i], 0))

            loss.append(sum(loss_each))

        loss = sum(loss)

        return loss / batch

    def predict(self, input_word_iv: Tensor, input_word_ooev: Tensor, input_char: Tensor, mask: Tensor) \
            -> Union[List[List[NestedSequenceLabel]], List[NestedSequenceLabel]]:
        # output from rnn [batch, length, tag_space]
        output = self._get_rnn_output(input_word_iv, input_word_ooev, input_char, mask=mask)

        batch, length, _ = output.size()

        preds = []

        for label in range(len(self.all_crfs)):
            preds_batch, energy_batch = self.all_crfs[label].decode(output, mask=mask)

            b_id = self.b_id
            i_id = self.i_id
            e_id = self.e_id
            o_id = self.o_id
            eos_id = self.eos_id
            decode_nest = self.all_crfs[label].decode_nest

            def predict_recursively(preds: Tensor, energy: Tensor, offset: int) -> NestedSequenceLabel:
                length = preds.size(0)
                nested_preds_list = []
                index = 0
                while index < length:
                    id = preds[index]
                    if id == eos_id:
                        break
                    if id != o_id:
                        if id == b_id:  # B-XXX
                            start_tmp = index
                            index += 1
                            if index == length:
                                break
                            id = preds[index]
                            while id == i_id:  # I-XXX
                                index += 1
                                if index == length:
                                    break
                                id = preds[index]
                            if id == e_id:  # E-XXX
                                end_tmp = index + 1
                                nested_preds = decode_nest(energy[start_tmp:end_tmp, :, :])
                                nested_preds_list.append(predict_recursively(nested_preds,
                                                                             energy[start_tmp:end_tmp, :, :],
                                                                             start_tmp + offset))
                    index += 1
                return NestedSequenceLabel(offset, length + offset, preds, nested_preds_list)

            preds_each = []
            for i in range(batch):
                preds_each.append(predict_recursively(preds_batch[i, :], energy_batch[i, :, :, :], 0))

            preds.append(preds_each)

        return preds
