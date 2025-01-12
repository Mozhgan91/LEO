# --------------------------------------------------------
# LEO
# Copyright (c) 2024 Waterloo's Wiselab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, vit_hidden_size, downsample_ratio, llm_hidden_size):
        super(MLP, self).__init__()
        # Explicitly add each layer so that we avoid double nesting
        self.add_module('0', nn.LayerNorm(vit_hidden_size * int(1 / downsample_ratio) ** 2))
        self.add_module('1', nn.Linear(vit_hidden_size * int(1 / downsample_ratio) ** 2, llm_hidden_size))
        self.add_module('2', nn.GELU()) 
        self.add_module('3', nn.Linear(llm_hidden_size, llm_hidden_size)) 

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

