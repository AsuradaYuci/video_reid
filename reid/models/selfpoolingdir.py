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

        # add lstm to generate q
    def forward(self, probe_value, probe_base):
        pro_size = probe_value.size()
        pro_batch = pro_size[0]
        pro_len = pro_size[1]

        # generating Querys
        # Qs = probe_base.view(pro_batch * pro_len, -1)
        # Qs = self.featQ(Qs)
        # Qs = self.featQ_bn(Qs)
        # Qs = Qs.view(pro_batch, pro_len, -1)
        #
        # Qmean = torch.mean(Qs, 1)
        # Qs = Qmean.squeeze(1)
        # Hs = Qmean.unsqueeze(1).expand(pro_batch, pro_len, self.output_num)

        # use LSTM generating Querys
        Qs = probe_base.view(pro_batch, pro_len, -1)
        Qs, (h_n, c_n) = self.lstm(Qs)  # QS = 4*8*128  h_n = 1*4*128 = Qs[,:-1,] c_n = 1*4*128
        Qs = h_n.squeeze(1)  # Qs = h_n = 1*4*128
        # Qs = Qs.permute(0, 2, 1)  # 4*128*10
        # Qs = F.avg_pool1d(Qs, pro_len)  # 4*128*1
        Qs = Qs.view(pro_batch, self.output_num)  # 4*128
        Qs = self.lstm_bn(Qs)

        Hs = Qs.unsqueeze(1).expand(pro_batch, pro_len, self.output_num)  # Hs = 4*8*128
        # generating Keys
        K = probe_base.view(pro_batch * pro_len, -1)
        K = self.featK(K)
        K = self.featK_bn(K)
        K = K.view(pro_batch, pro_len, -1)  # 4*8*128

        weights = Hs * K
        weights = weights.permute(0, 2, 1)
        weights = weights.contiguous()
        weights = weights.view(-1, pro_len)
        weights = self.softmax(weights)
        weights = weights.view(pro_batch, self.output_num, pro_len)
        weights = weights.permute(0, 2, 1)

        pool_probe = probe_value * weights
        pool_probe = pool_probe.sum(1)
        pool_probe = pool_probe.squeeze(1)

        # pool_probe = torch.mean(probe_value, 1)
        # pool_probe = pool_probe.squeeze(1)
        #

        # pool_probe  Batch x featnum
        # Hs  Batch x hidden_num

        return pool_probe, Qs
