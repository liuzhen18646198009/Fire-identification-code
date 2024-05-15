# -- coding: utf-8 --
import torch
import torchvision
from thop import profile
from nets.CSPdarknet import CSPDarknet
# Model
print('==> Building model..')
model = CSPDarknet()

dummy_input = torch.randn(1, 3, 416, 416)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

