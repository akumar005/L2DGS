import torch
import torch.nn as nn
import torch.nn.functional as F


class PointReflectace(nn.Module):
    def __init__(self, D=3, W=64, input_ch=3, input_ch_views=3, output_ch=3,embed_fn=None, \
                 output_color_ch=3,):
        super(PointReflectace, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
    

        layers = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            layer = nn.Linear

            in_channels = W

            layers += [layer(in_channels, W)]

        self.pts_linears = nn.ModuleList(layers)
       
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, h):
        
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
        
        outputs = self.output_linear(h)

        return outputs


