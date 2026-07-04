import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np


class LiftIntensity(nn.Module):
    def __init__(self):
        super(LiftIntensity,self).__init__()
        
        
    def forward(self,image_tensor):
        '''
        image_tensor: Bx3xHxW 
        '''
        
        out = 0.5 - torch.sin(torch.asin(1-2*image_tensor)/3)  
        return out


class Luminance_score(nn.Module):
    def __init__(self):
        super(Luminance_score,self).__init__()
        self.luminance_weight = torch.tensor([0.2126, 0.7152, 0.0722],dtype=torch.float32)

    def forward(self,x):
        luminance_weight = self.luminance_weight.to(x.device)
        score = torch.sum(x * luminance_weight.view(1,3,1,1),axis=1)

        return torch.mean(score)



class successive_smoothness(nn.Module):
    def __init__(self,del_d=1.0):
        super(successive_smoothness,self).__init__()
        self.del_d = del_d

    def forward(self,f1,f2):
        out = f2 - self.del_d*f1
        return torch.norm(out)
        

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

			
class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E


class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d
        
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

        
class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)
    def forward(self, x ):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b,c,h,w = x.shape
        # x_de = x.cpu().detach().numpy()
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        # print(k)
        

        k = torch.mean(k)
        return k

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3


class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def forward(self, est_depth, gt_disp):
        """
        Compute scale-invariant depth loss.

        Args:
            est_depth (torch.Tensor): Estimated depth map of shape (N, 1, H, W).
            gt_disp (torch.Tensor): Ground truth disparity map of shape (N, 1, H, W).

        Returns:
            torch.Tensor: Scale-invariant depth loss.
        """
        # Convert ground truth disparity to depth by inverting
        gt_depth = 1.0 / (gt_disp + 1e-8)

        # Normalize the estimated depth and ground truth depth to [-0.5, 0.5]
        est_depth_norm = (est_depth - est_depth.mean(dim=(1, 2, 3), keepdim=True)) / (est_depth.std(dim=(1, 2, 3), keepdim=True) + 1e-8)
        gt_depth_norm = (gt_depth - gt_depth.mean(dim=(1, 2, 3), keepdim=True)) / (gt_depth.std(dim=(1, 2, 3), keepdim=True) + 1e-8)

        # Compute the difference
        diff = est_depth_norm - gt_depth_norm

        # Scale-invariant loss
        loss = torch.mean(diff ** 2) #- 0.5 * torch.mean(diff) ** 2

        return loss



class SmoothLaplacian(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super(SmoothLaplacian, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.smoothing = self.get_gaussian_kernel(kernel_size, sigma)
        
        # Laplacian kernel
        self.laplacian_kernel = torch.tensor([
            [0,  1,  0],
            [1, -4,  1],
            [0,  1,  0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape (1,1,3,3)

    def get_gaussian_kernel(self, kernel_size, sigma):
        """Generates a 2D Gaussian kernel."""
        x = torch.arange(kernel_size) - kernel_size // 2
        y = torch.arange(kernel_size) - kernel_size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape (1,1,K,K)
        return kernel
    
    def forward(self, x):
        n, c, h, w = x.shape
        device = x.device
        
        # Apply Gaussian smoothing to each channel separately
        smoothed = F.conv2d(x.view(n * c, 1, h, w), self.smoothing.to(device), padding=self.kernel_size//2, groups=1)
        smoothed = smoothed.view(n, c, h, w)
        
        # Apply Laplacian filter to each channel separately
        laplacian = F.conv2d(smoothed.view(n * c, 1, h, w), self.laplacian_kernel.to(device), padding=1, groups=1)
        laplacian = laplacian.view(n, c, h, w)
        
        return laplacian


class GammaLog(nn.Module):
    def __init__(self, kernel_size=3, sigma=0.3, threshold=0.0, sharpness=50):
        super(GammaLog, self).__init__()
        self.model = SmoothLaplacian(kernel_size=kernel_size, sigma=sigma)
        self.threshold = threshold
        self.sharpness = sharpness

    def soft_threshold(self, tensor):
        return torch.sigmoid(self.sharpness * (self.threshold - tensor))

    def forward(self, image_tensor):
        output = self.model(image_tensor)   
        output = self.soft_threshold(output)
        final_op = image_tensor + output
        #final = 1.0 - final
        
        return final_op



class ImageGradient(nn.Module):
    def __init__(self):
        """
        This module computes the per-channel gradient magnitude for an RGB image.
        It uses fixed finite difference kernels to compute the horizontal and vertical gradients.
        """
        super(ImageGradient, self).__init__()
        
        # Define a horizontal derivative kernel: [-1, 0, 1]
        kernel_x = torch.tensor([[-1., 0., 1.]], dtype=torch.float32)  # shape: (1, 3)
        kernel_x = kernel_x.view(1, 1, 1, 3)  # reshape to (out_channels=1, in_channels=1, H=1, W=3)
        
        # Define a vertical derivative kernel: transpose of [-1, 0, 1]
        kernel_y = torch.tensor([[-1.], [0.], [1.]], dtype=torch.float32)  # shape: (3, 1)
        kernel_y = kernel_y.view(1, 1, 3, 1)  # reshape to (1, 1, H=3, W=1)
        
        # We need one kernel per channel. Using groups convolution, we replicate each kernel for 3 channels.
        # Final shapes will be (3, 1, kernel_height, kernel_width)
        self.register_buffer('kernel_x', kernel_x.repeat(3, 1, 1, 1))
        self.register_buffer('kernel_y', kernel_y.repeat(3, 1, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input RGB image tensor of shape (B, 3, H, W)
            
        Returns:
            torch.Tensor: Per-channel gradient magnitude tensor of shape (B, 3, H, W)
        """
        # Use groups convolution to compute per-channel derivatives.
        # For kernel_x (1x3), pad only horizontally (pad width=1).
        dx = F.conv2d(x, self.kernel_x, padding=(0, 1), groups=x.shape[1])
        # For kernel_y (3x1), pad only vertically (pad height=1).
        dy = F.conv2d(x, self.kernel_y, padding=(1, 0), groups=x.shape[1])
        
        # Compute gradient magnitude. A small epsilon is added for numerical stability.
        grad_magnitude = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        return grad_magnitude



class VariancePooling(nn.Module):
    def __init__(self, window_size: int, stride: int = 1):
        super().__init__()
        self.window_size = window_size
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.shape
        pad = self.window_size // 2  # Padding to maintain spatial size

        # Pad input tensor (reflect padding prevents boundary artifacts)
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')

        # Extract sliding windows using unfold
        unfolded = F.unfold(x_padded, kernel_size=self.window_size, stride=self.stride)
        
        # Reshape to (batch, channels, window_size * window_size, num_patches)
        unfolded = unfolded.view(b, c, self.window_size * self.window_size, -1)

        # Compute variance along the window dimension
        var = unfolded.var(dim=2, unbiased=False)  # (b, c, num_patches)

        # Compute output spatial dimensions
        h_out = (h + 2 * pad - self.window_size) // self.stride + 1
        w_out = (w + 2 * pad - self.window_size) // self.stride + 1

        # Reshape back to (b, c, h', w')
        var = var.view(b, c, h_out, w_out)

        return var


class DepthSmoothLoss(torch.nn.Module):
    def __init__(self):
        super(DepthSmoothLoss, self).__init__()
        self.niter = 5
        self.emin = 0.01
        self.wg = 1.0e-2  # Weight for smoothness loss
        self.l1 = torch.nn.L1Loss()

    def gradient_x(self, img):
        kernel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        return torch.cat([F.conv2d(img[:, i, :, :].unsqueeze(1), kernel, padding=1) for i in range(img.shape[1])], dim=1)

    def gradient_y(self, img):
        kernel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        return torch.cat([F.conv2d(img[:, i, :, :].unsqueeze(1), kernel, padding=1) for i in range(img.shape[1])], dim=1)

    def edge_aware_smoothness_per_pixel(self, img, pred):
        """
        Computes edge-aware smoothness loss by comparing gradients of predicted depth/disparity and the input image.
        
        Args:
            img (tensor): RGB image of shape (B, 3, H, W)
            pred (tensor): Predicted depth/disparity of shape (B, 1, H, W)
        
        Returns:
            tensor: Scalar smoothness loss
        """
        pred_gradients_x = self.gradient_x(pred)
        pred_gradients_y = self.gradient_y(pred)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), dim=1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), dim=1, keepdim=True))
        
        smoothness_x = torch.abs(pred_gradients_x) * weights_x
        smoothness_y = torch.abs(pred_gradients_y) * weights_y

        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    def forward(self, img, pred):
        return self.edge_aware_smoothness_per_pixel(img, pred)

