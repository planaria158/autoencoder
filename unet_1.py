"""
Mark Thompson.  UNet 
https://arxiv.org/pdf/1807.10165.pdf
https://arxiv.org/pdf/1912.05074v2.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Set bias=False in the Conv2d layers when using BatchNorm
#
def conv_block(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def build_last_conv_bn_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.Sigmoid(),  #nn.ReLU(),
    )


class UNet_1(nn.Module):
    def __init__(self):
        super(UNet_1, self).__init__()
        
        nb_filter = [32, 64, 128, 256, 512, 1024]
        # self.num_classes = 1

        # Could have used nn.ConvTranspose2d instead of Upsample
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.pool = nn.MaxPool2d(2, 2)
        
        # input has 3 channels
        self.enc_block_0 = conv_block(in_channels=3, out_channels=nb_filter[0])     
        self.enc_block_1 = conv_block(nb_filter[0], nb_filter[1])
        self.enc_block_2 = conv_block(nb_filter[1], nb_filter[2])
        self.enc_block_3 = conv_block(nb_filter[2], nb_filter[3])
        self.enc_block_4 = conv_block(nb_filter[3], nb_filter[4])
        self.enc_block_5 = conv_block(nb_filter[4], nb_filter[5])
            
        self.decode_block_4 = conv_block(nb_filter[5] + nb_filter[4], nb_filter[4])        
        self.decode_block_3 = conv_block(nb_filter[4] + nb_filter[3], nb_filter[3])        
        self.decode_block_2 = conv_block(nb_filter[3] + nb_filter[2], nb_filter[2])        
        self.decode_block_1 = conv_block(nb_filter[2] + nb_filter[1], nb_filter[1])        
        self.decode_block_0 = conv_block(nb_filter[1] + nb_filter[0], nb_filter[0])
        
        # last layer re-creates the rgb image
        self.last = build_last_conv_bn_block(nb_filter[0], 3)   #self.num_classes)
        

    def forward(self, x):
        # Encoder
        enc_0 = self.enc_block_0(x)        
        enc_1 = self.enc_block_1(self.pool(enc_0))
        enc_2 = self.enc_block_2(self.pool(enc_1))        
        enc_3 = self.enc_block_3(self.pool(enc_2))        
        enc_4 = self.enc_block_4(self.pool(enc_3))

        # the "Center"        
        enc_5 = self.enc_block_5(self.pool(enc_4))

        # Decoder
        dec_4 = self.decode_block_4(torch.cat([enc_4, self.up(enc_5)],1))
        dec_3 = self.decode_block_3(torch.cat([enc_3, self.up(dec_4)],1))      
        dec_2 = self.decode_block_2(torch.cat([enc_2, self.up(dec_3)],1))
        dec_1 = self.decode_block_1(torch.cat([enc_1, self.up(dec_2)],1))
        dec_0 = self.decode_block_0(torch.cat([enc_0, self.up(dec_1)],1))
 
        last = self.last(dec_0)
    
        return last #torch.sigmoid(last) 
