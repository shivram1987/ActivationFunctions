import warnings
from typing import Tuple, Optional

import torch
from torch import Tensor
#from .linear import _LinearWithBias
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
#from .module import Module
#from torch.nn.Module import Module
#from .. import functional as F
import torch.nn.functional as F

class Swish(torch.nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        self.num_parameters = num_parameters
        super(Swish, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, input: Tensor) -> Tensor:
        return input * F.sigmoid(self.weight * input)

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)
		
    #def forward(self, input: Tensor) -> Tensor:
    #    return input * torch.sigmoid(input)
		


class Elliott(torch.nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(Elliott, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input / (1 + torch.abs(input)) + 0.5

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str



class ABReLU(torch.nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ABReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        #print(input.shape)
        input1 = torch.mean(input,dim=(0,1,2,3))
        #input1 = torch.mean(input.view(input.size(0), -1), dim=2)
        #input1 = F.adaptive_avg_pool2d(input, (1, 1, 1))
        #print(input1.shape)
        #print(input1)
        #print(input1(1,1,1,1:10))
        input = input - input1
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str



class LiSHT(torch.nn.Module):
     def forward(self, input: Tensor) -> Tensor:
        return input * torch.tanh(input)


class Mish(torch.nn.Module):
     def forward(self, input: Tensor) -> Tensor:
        return input * F.tanh(F.softplus(input))




class SRS(torch.nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int

#    def __init__(self, num_parameters: int = 1, init: float = (5.0,3.0)) -> None:
    def __init__(self, num_parameters: int = 1, init: float = (10.0,10.0)) -> None:  #for SENet18
        self.num_parameters = num_parameters
        super(SRS, self).__init__()
        self.weight1 = Parameter(torch.Tensor(num_parameters).fill_(init[0]))
        self.weight2 = Parameter(torch.Tensor(num_parameters).fill_(init[1]))

    def forward(self, input: Tensor) -> Tensor:
        #self.weight1 = torch.abs(self.weight1)
        #self.weight2 = torch.abs(self.weight2)
        #w1 = self.weight1 + 1e-8
        #print(w1.shape)
        #print(input.shape)
        #a = torch.div(input,abs(self.weight1+1e-8))
        #b = torch.exp(- torch.div(input,abs(self.weight2+1e-8)))
        #return torch.div(input, a + b + 1e-8)
        return torch.div(input, 1e-2 + torch.div(input,torch.abs(self.weight1)+1e-2) + torch.exp(-torch.div(input,torch.abs(self.weight2)+1e-2)))

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)







class PDELU(torch.nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 1.0) -> None:
        self.num_parameters = num_parameters
        super(PDELU, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, input: Tensor) -> Tensor:
        input1 = input
        input1[input < 0] = 0
        input2 = self.weight * (torch.pow(1+0.1*input, 10) - 1)
        input2[input >= 0] = 0
        return input1 + input2
        
    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)














