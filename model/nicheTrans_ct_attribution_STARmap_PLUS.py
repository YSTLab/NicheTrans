from __future__ import absolute_import

import torch
import torchvision
from torch import nn

from model.attention import *
from model.nicheTrans import *


class NetBlock(nn.Module):
    def __init__(
        self,
        nlayer: int, 
        dim_list: list, 
        dropout_rate: float, 
        noise_rate: float
        ):

        super(NetBlock, self).__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        
        for i in range(nlayer):      
            self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            nn.init.xavier_uniform_(self.linear_list[i].weight)
            self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
            self.activation_list.append(nn.LeakyReLU())
            if not i == nlayer -1: 
                self.dropout_list.append(nn.Dropout(dropout_rate))
        
    def forward(self, x):
        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            x = self.linear_list[i](x)
            x = self.bn_list[i](x)
            x = self.activation_list[i](x)
            if not i == self.nlayer -1:
                """ don't use dropout for output to avoid loss calculate break down """
                x = self.dropout_list[i](x)
        return x
    

class NicheTrans_ct(nn.Module):
    # def __init__(self, rna_length=877, msi_length=137):
    def __init__(self, source_length=877, target_length=137, noise_rate=0.2, dropout_rate=0.1):
        super(NicheTrans_ct, self).__init__()

        self.source_length, self.target_length = source_length, target_length
        self.noise_rate, self.dropout_rate = noise_rate, dropout_rate

        self.fea_size, self.img_size = 256, 256

        self.encoder_rna = NetBlock(nlayer=2, dim_list=[source_length, 512, self.fea_size], dropout_rate=self.dropout_rate, noise_rate=self.noise_rate)

        self.projection_rna = nn.Sequential(nn.Linear(256, 256),
                                            nn.LayerNorm(256),
                                            nn.ReLU(inplace=True))

        self.fusion_omic = Self_Attention(query_dim=self.fea_size, context_dim=self.fea_size, heads=4, dim_head=64, dropout=self.dropout_rate)
        self.ffn_omic = FeedForward(dim=self.fea_size, mult=2)

        self.ln1 = nn.LayerNorm(self.fea_size)
        self.ln2 = nn.LayerNorm(self.fea_size)
        
        ################
        self.dropout = nn.Dropout(self.dropout_rate)

        # define the prediction layer
        predict_net = []
        for _ in range(target_length):
            predict_net.append(
                nn.Sequential(nn.Linear(self.fea_size, 128),
                             nn.BatchNorm1d(128),
                             nn.LeakyReLU(),
                            #  nn.Linear(128, 1, bias=True))
                            nn.Linear(128, 1, bias=False))
            )
        self.predict_layers = nn.ModuleList(predict_net)

        ###############

        # to define and normalize the tokens
        self.token_center = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.token_neigh_1 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.token_neigh_2 = nn.Parameter(torch.randn((1, 1, self.fea_size), requires_grad=True))
        self.cell_tokens = nn.Parameter(torch.randn((1, 1, 13, self.fea_size), requires_grad=True))

        trunc_normal_(self.cell_tokens, std=.02)
        trunc_normal_(self.token_center, std=.02)
        trunc_normal_(self.token_neigh_1, std=.02)
        trunc_normal_(self.token_neigh_2, std=.02)
       
    def forward(self, input):
        b = input.size(0)

        cell_inf = input[:, :, -13:]
        omics_data = input[:, :, 0: -13]

        classes_tokens = (self.cell_tokens * cell_inf.unsqueeze(dim=-1)).sum(-2)

        spatial_tokens = torch.cat([self.token_center, self.token_neigh_1.repeat(1, 6, 1), self.token_neigh_2.repeat(1, 6, 1)], dim=1)

        omic_data = omics_data.view(-1, self.source_length)

        # genome feature extraction, be aware that we add on the features
        f_omic = self.encoder_rna(omic_data).view(b, -1, self.fea_size) 
        f_omic = f_omic + spatial_tokens + classes_tokens

        f_omic = self.projection_rna(f_omic)

        f_omic = self.fusion_omic(self.ln1(f_omic)) + f_omic
        f_omic = self.ffn_omic(self.ln2(f_omic)) + f_omic

        f_omic = f_omic[:, 0, :]
        f = self.dropout(f_omic)

        # final prediction
        # out = []
        # for i in range(self.target_length):
        #     out.append(self.predict_layers[i](f))
        # out = torch.cat(out, dim=1)
        out = self.predict_layers[1](f)

        return out