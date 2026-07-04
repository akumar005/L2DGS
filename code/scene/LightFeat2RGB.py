import torch
import torch.nn as nn
import torch.nn.init as init

class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()

        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(2, 8)
        self.relu1 = nn.LeakyReLU(0.01)

       
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(4, 16)
        self.relu2 = nn.LeakyReLU(0.01)

       
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(2, 8)
        self.relu3 = nn.LeakyReLU(0.01)

        
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=1)

       
        self._initialize_weights()

    def forward(self, x):
        x = torch.tanh(x)  # Normalize input to [-1, 1]

       
        x1 = self.relu1(self.gn1(self.conv1(x)))

        
        x2 = self.relu2(self.gn2(self.conv2(x1)))

        
        x3 = self.gn3(self.conv3(x2)) + 0.3 * x1
        x3 = self.relu3(x3)

        
        trans_out = self.conv4(x3)


       
        trans_out = trans_out / (1 + torch.abs(trans_out)) * 4.5 + 5.5

        
        return trans_out #torch.cat([rgb_out, trans_out], dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def get_cnn_parameters(self):
        return self.parameters()
