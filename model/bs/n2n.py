import torch
import torch.nn as nn

class CBLK(nn.Module):
    def __init__(self, inChannels, outChannels, k = 3, s = 1, p = 1, bias = True) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size = 3, stride=s, padding = p,  bias = bias),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
    
class UCC(nn.Module):
    def __init__(self, inChannels = [144, 96], outChannels = [96, 96]) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)

        self.cblk = nn.Sequential(
            CBLK(inChannels[0], outChannels[0]),
            CBLK(inChannels[1], outChannels[1]),
        )
    def forward(self, x, n):
        n = torch.cat([self.up(x), n], 1)
        return self.cblk(n)
    
class Noise2Noise(nn.Module):
    def __init__(self, inChannel = 1, outChannel = 1) -> None:
        super().__init__()
        self.enc_0 = nn.Sequential(
            CBLK(inChannel, 48),
            CBLK(48, 48),
            nn.MaxPool2d(2),
        )
        self.enc_1 = nn.Sequential(
            CBLK(48, 48),
            nn.MaxPool2d(2),
        )
        self.enc_2 = nn.Sequential(
            CBLK(48, 48),
            nn.MaxPool2d(2),
        )
        self.enc_3 = nn.Sequential(
            CBLK(48, 48),
            nn.MaxPool2d(2),
        )

        self.enc_4 = nn.Sequential(
            CBLK(48, 48),
            nn.MaxPool2d(2),
            CBLK(48, 48),
        )

        self.dec_0 = UCC([96 + inChannel, 64], [64, 32])
        self.dec_1 = UCC()
        self.dec_2 = UCC()
        self.dec_3 = UCC()
        self.dec_4 = UCC([96 , 96], [96, 96])

        self.dec_out = nn.Conv2d(32, outChannel,1,1)



    def forward(self, x):
        skips = [x.clone().detach()]

        n = self.enc_0(x) 
        skips.append(n)

        n = self.enc_1(n)
        skips.append(n)

        n = self.enc_2(n)
        skips.append(n)

        n = self.enc_3(n)
        skips.append(n)

        # ..............
        n = self.enc_4(n)
        # ..............

        n = self.dec_4(n, skips.pop())
        # print(n.shape, skips[-1].shape)
        n = self.dec_3(n, skips.pop())
        n = self.dec_2(n, skips.pop())
        n = self.dec_1(n, skips.pop())
        n = self.dec_0(n, skips.pop())
        n = self.dec_out(n)

        return n
