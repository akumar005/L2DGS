# import torch
# import torch.nn as nn


# class PointMEstimator(nn.Module):
#     def __init__(self, input_feat=128, num_filters=64, output_res=16):
#         """
#         Neural network to estimate M matrix from input features.

#         Args:
#             input_feat (int): Dimension of the input feature.
#             num_filters (int): Number of filters in the CNN layers.
#             output_res (int): Resolution of the final output (e.g., 16x16).
#         """
#         super(PointMEstimator, self).__init__()

#         self.input_feat = input_feat
#         self.num_filters = num_filters
#         self.output_res = output_res

#         # MLP to process input features
#         self.mlp = nn.Sequential(
#             nn.Linear(self.input_feat, self.num_filters),
#             nn.ELU(),
#         )

#         # CNN for spatial processing
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, self.num_filters, kernel_size=3, stride=1, padding=1),
#             nn.ELU(),
#             nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, stride=1, padding=1),
#             nn.ELU(),
#         )

#         # Upsample to the target resolution
#         self.upsample = nn.ConvTranspose2d(
#             self.num_filters, self.num_filters, kernel_size=2, stride=2
#         )

#         # Final layer to produce 3-channel (RGB) output
#         self.final_layer = nn.Conv2d(
#             self.num_filters, 3, kernel_size=3, stride=1, padding=1
#         )

#     def forward(self, x):
#         """
#         Forward pass to estimate M.

#         Args:
#             x (torch.Tensor): Input features of shape (batch_size, input_feat).

#         Returns:
#             torch.Tensor: Estimated M matrices of shape (batch_size, 3, output_res, output_res).
#         """
#         batch_size = x.shape[0]

#         # Process input features through MLP
#         x = self.mlp(x)  # (batch_size, num_filters)

#         # Reshape for CNN input
#         x = x.view(batch_size, 1, 8, 8)  # Assumes input feature maps to 8x8 spatial size

#         # Pass through CNN layers
#         x = self.cnn(x)

#         # Upsample to target resolution
#         x = self.upsample(x)  # (batch_size, num_filters, output_res, output_res)

#         # Final layer to produce 3-channel output
#         x = self.final_layer(x)  # (batch_size, 3, output_res, output_res)

#         return x


# # if __name__ == "__main__":
# #     # Instantiate the model
# #     model = PointMEstimator()
# #     model = model.cuda()

# #     # Example input feature
# #     x = torch.rand(5, 128).cuda()

# #     # Forward pass
# #     out = model(x)

# #     # Output shape
# #     print("Output Shape:", out.shape)  # Expected: (batch_size, 3, 16, 16)



import torch
import torch.nn as nn


class WellSHEstimator(nn.Module):
    def __init__(self,feat_inp=128):
        super(WellSHEstimator,self).__init__()

        self.feat_inp = feat_inp

        self.common = nn.Linear(self.feat_inp , 32)
        self.relu1  = nn.ReLU()

        self.l0 = nn.Linear(32,3)
        self.sigmoid0 = nn.Sigmoid()

        self.l1 = nn.Linear(32,9)
        self.sigmoid1 = nn.Sigmoid()

        self.l2 = nn.Linear(32,15)
        self.sigmoid2 = nn.Sigmoid()

        self.l3 = nn.Linear(32,21)
        self.sigmoid3 = nn.Sigmoid()

    def forward(self,x):
        '''
        x: feat of shape (N,128)
        '''

        x = self.relu1(self.common(x))


        l0_rgb = self.sigmoid0(self.l0(x))*2 - 1
        l1_rgb = self.sigmoid1(self.l1(x))*2 - 1
        l2_rgb = self.sigmoid2(self.l2(x))*2 - 1
        l3_rgb = self.sigmoid3(self.l3(x))*2 - 1

        l0_rgb = l0_rgb.view(-1,1,3)
        l1_rgb = l1_rgb.view(-1,3,3)
        l2_rgb = l2_rgb.view(-1,5,3)
        l3_rgb = l3_rgb.view(-1,7,3)

        out = torch.cat((l0_rgb, l1_rgb, l2_rgb, l3_rgb),dim=1)
        
        return out



# if __name__ == '__main__':
#     model = WellSHEstimator(128).cuda()
#     x = torch.rand(200,128).cuda()
#     y = model(x)

#     print(y.shape)
