from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.init as init
from torch.nn import functional as F


class SelfPoolingDir(nn.Module):
    def __init__(self, input_num, output_num):
        super(SelfPoolingDir, self).__init__()
        self.input_num = input_num
        self.output_num = output_num

        # todo: LSTM
        self.lstm = nn.LSTM(input_size=self.input_num,
                            hidden_size=self.output_num, num_layers=1, batch_first=True, dropout=0)
        self.bilstm = nn.LSTM(input_size=self.input_num, hidden_size=self.output_num,
                              num_layers=1, batch_first=True, dropout=0, bidirectional=True)
        self.lstm_bn = nn.BatchNorm1d(self.output_num)

        ## Linear K
        self.featK = nn.Linear(self.input_num, self.output_num)
        self.featK_bn = nn.BatchNorm1d(self.output_num)

        ## Linear_Q
        self.featQ = nn.Linear(self.input_num, self.output_num)
        self.featQ_bn = nn.BatchNorm1d(self.output_num)


        ## Softmax
        self.softmax = nn.Softmax(dim=-1)

        init.kaiming_uniform_(self.featK.weight, mode='fan_out')
        init.constant_(self.featK.bias, 0)

        init.constant_(self.featK_bn.weight, 1)
        init.constant_(self.featK_bn.bias, 0)

        init.kaiming_uniform_(self.featQ.weight, mode='fan_out')
        init.constant_(self.featQ.bias, 0)
        init.constant_(self.featQ_bn.weight, 1)
        init.constant_(self.featQ_bn.bias, 0)
        init.constant_(self.lstm_bn.weight, 1)
        init.constant_(self.lstm_bn.bias, 0)

        # add lstm to generate q
    def forward(self, probe_value, probe_base):
        pro_size = probe_value.size()  # torch.Size([6, 8, 2048])
        pro_batch = pro_size[0]
        pro_len = pro_size[1]

        # use LSTM generating Querys
        Qs = probe_base.view(pro_batch, pro_len, -1)  # 6 * 8 * 2048
        Qs, (h_n, c_n) = self.bilstm(Qs)  # QS = 6*8*128  h_n = 2*6*128 = Qs[,:-1,] c_n = 2*6*128
        Qs = h_n[0, :, :] + h_n[1, :, :]  # Qs = h_n = 6*128
        # Qs, (h_n, c_n) = self.lstm(Qs)
        # Qs = h_n.squeeze(1)
        Qs = Qs.view(pro_batch, self.output_num)  # 6*128
        Qs = self.lstm_bn(Qs)

        Hs = Qs.unsqueeze(1).expand(pro_batch, pro_len, self.output_num)  # Hs = 6*8*128
        # generating Keys
        K = probe_base.view(pro_batch * pro_len, -1)  # torch.Size([48, 2048])
        K = self.featK(K)
        K = self.featK_bn(K)
        K = K.view(pro_batch, pro_len, -1)  # 6*8*128

        weights = Hs * K  # torch.Size([6, 8, 128])
        weights = weights.permute(0, 2, 1)  # torch.Size([6, 128, 8])
        weights = weights.contiguous()
        weights = weights.view(-1, pro_len)  # torch.Size([768, 8])
        weights = self.softmax(weights)
        weights = weights.view(pro_batch, self.output_num, pro_len)  # torch.Size([6, 128, 8])
        weights = weights.permute(0, 2, 1)  # torch.Size([6, 8, 128])

        pool_probe = probe_value * weights  # torch.Size([6, 8, 128])
        pool_probe = pool_probe.sum(1)  # torch.Size([6, 128])
        pool_probe = pool_probe.squeeze(1)

        # pool_probe = torch.mean(probe_value, 1)
        # pool_probe = pool_probe.squeeze(1)
        #

        # pool_probe  Batch x featnum
        # Hs  Batch x hidden_num

        return pool_probe, Qs
