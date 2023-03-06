#--------------------------------------------#
#  This part of the code is used to look at the network structure
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.yolo import YoloBody

if __name__ == "__main__":
    input_shape = [640, 640]
    num_classes = 80
    phi         = 'l'
    
    # You need to use Device to specify the network to run in the GPU or CPU
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m       = YoloBody(num_classes, phi).to(device)
    summary(m, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
