import torch
import torch.nn as nn

class TimeConv(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        
        super().__init__()
        
        self.conv = nn.Conv2d(c_in + 1, c_out, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, t):

        batch_size, _, height, width = x.shape
        time_embed = t * torch.ones(batch_size, 1, height, width).to(x.device)
        x_embed = torch.cat([x, time_embed], dim=1)
        return self.conv(x_embed)