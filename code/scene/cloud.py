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
    def __init__(self, feat_inp=48):
        super(WellSHEstimator, self).__init__()

        self.feat_inp = feat_inp

        # Define layers
        self.linear1 = nn.Linear(self.feat_inp, 32)
        self.leaky1 = nn.ReLU()

        self.linear2 = nn.Linear(32, 64)
        self.leaky2 = nn.ReLU()

        self.linear3 = nn.Linear(64, 64)
        self.leaky3 = nn.ReLU()

        self.linear4 = nn.Linear(64, 48)
        #self.leaky4 = nn.ReLU()

        #self.linear5 = nn.Linear(64, 48)

        # Initialize poc_buf
        self.poc_buf = torch.FloatTensor([(2**i) for i in range(4)])

        # Apply initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights of the network."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        x: feat of shape (N, 128)
        '''
        x = x.view(x.shape[0], -1)
        #x = poc_fre(x, self.poc_buf.to(x.device))

        x = self.leaky1(self.linear1(x))
        x = self.leaky2(self.linear2(x))
        x = self.leaky3(self.linear3(x))
        x = self.linear4(x)
        #x = self.linear5(x)

        # x = 2 * x - 1  # x in [-1, 1]

        return x.view(-1, 16, 3)




def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb



# if __name__ == '__main__':
#     model = WellSHEstimator().cuda()
#     x = torch.rand(200,16,3).cuda()
#     y = model(x)

#     print(y.shape)
