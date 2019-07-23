import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True):
        super(_NonLocalBlockND, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        conv_nd = nn.Conv2d
        # self.max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        self.max_pool_layer = nn.AdaptiveAvgPool2d(10)
        bn = nn.BatchNorm2d


        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)


        self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, self.max_pool_layer)
            self.phi = self.max_pool_layer

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            g_x = self.max_pool_layer(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            g_x = x.view(batch_size, self.inter_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


if __name__ == '__main__':
    import torch
    import cv2
    from PIL import Image
    import torchvision.transforms as transforms

    transform1 = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    ]
    )
    # img = cv2.imread('hasky.jpg',cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread('hasky.jpg')
    img = Image.open('hasky.jpg')
    img = transform1(img)
    img = img.unsqueeze(0)
    net = _NonLocalBlockND(in_channels=1,sub_sample=True)
    out = net(img)
    print(out.size())


