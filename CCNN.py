#les imports necessaires
import torch.nn as nn
import torch
import gc
from complex_translator import real3D,imag3D,make_complex
#sur des données en covention pytorch de forme :
# [batch,channels,d1,d2,d3,complex]

def CReLU(x):
	x_real = nn.functional.relu(real3D(x))
	x_imag = nn.functional.relu(imag3D(x))
	return make_complex(x_real,x_imag)

class CCNN(nn.Module):
	def __init__(self,channels_in,channels_out,s,kernel_size,padding,bias=False):
		super(CCNN, self).__init__()
		self.convRealLayer = nn.Conv3d(channels_in,channels_out,kernel_size,stride=s,padding=padding,bias=bias)#la partie réelle du vecteur de conv
		self.convImagLayer = nn.Conv3d(channels_in,channels_out,kernel_size,stride=s,padding=padding,bias=bias)#la partie imaginaire du vecteur de conv

	def forward(self,x,device=None):
		# avec W = A+iB, h= x+iy:
		# W*h = (A * x - B * y) + i ( B * x + A * y)   *:opération de convolution, 
		real_x = real3D(x)
		imag_x = imag3D(x)
		if device is not None:
			real_x=real_x.to(device)
			imag_x=imag_x.to(device)
			self.convRealLayer=self.convRealLayer.to(device)
			self.convImagLayer=self.convImagLayer.to(device)
		x = None #on ne se sert plus de x : liberer
		real_part = self.convRealLayer(real_x) - self.convImagLayer(imag_x) #A*x - B*y
		imag_part = self.convImagLayer(real_x) +self.convRealLayer(imag_x)  #B*x + A*y
		return make_complex(real_part, imag_part)


class CCNNTranspose(CCNN):
	def __init__(self,channels_in,channels_out,kernel_size,s=(1,1,1),padding=(0,0,0),bias=False):
		super(CCNN, self).__init__()
		self.convRealLayer = nn.ConvTranspose3d(channels_in,channels_out,kernel_size,stride=s,padding=padding,bias=bias)#la partie réelle du vecteur de conv
		self.convImagLayer = nn.ConvTranspose3d(channels_in,channels_out,kernel_size,stride=s,padding=padding,bias=bias)#la partie imaginaire du vecteur de conv