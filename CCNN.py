#les imports necessaires
import torch
from complex_translator import real3D,imag3D,make_complex
#sur des données en covention pytorch de forme :
# [batch,channels,d1,d2,d3,complex]

#implémentation de http://arxiv.org/abs/1705.09792

#une fonction de 
def CReLU(x):
	x_real = torch.nn.functional.relu(real3D(x))
	x_imag = torch.nn.functional.relu(imag3D(x))
	return make_complex(x_real,x_imag)

class CCNN(nn.Module):
	def __init__(self,channels_in,channels_out,s,kernel_size,padding,bias=False):
		super(CCNN, self).__init__()
		self.convRealLayer = torch.nn.Conv3d(channels_in,channels_out,kernel_size,stride=s,padding=padding,bias=bias)#la partie réelle du vecteur de conv
		self.convImagLayer = torch.nn.Conv3d(channels_in,channels_out,kernel_size,stride=s,padding=padding,bias=bias)#la partie imaginaire du vecteur de conv
	def to(self,device):
		self.convRealLayer=self.convRealLayer.to(device)
		self.convImagLayer=self.convImagLayer.to(device)
	
	def forward(self,x):
		# avec W = A+iB, h= x+iy:
		# W*h = (A * x - B * y) + i ( B * x + A * y)   *:opération de convolution, 
		real_x = real3D(x) # extraire la partie réelle de x
		imag_x = imag3D(x) # extraire la partie imaginaire de x
		x = None #on ne se sert plus de x : liberer, devrait se faire avec garbage collector mais on le force
		real_part = self.convRealLayer(real_x) - self.convImagLayer(imag_x) #A*x - B*y
		imag_part = self.convImagLayer(real_x) +self.convRealLayer(imag_x)  #B*x + A*y
		return make_complex(real_part, imag_part) #on créé un tenseur complexe dans la convention pytorch


class CCNNTranspose(CCNN):
	def __init__(self,channels_in,channels_out,kernel_size,s=(1,1,1),padding=(0,0,0),bias=False):
		super(CCNN, self).__init__()
		self.convRealLayer = torch.nn.ConvTranspose3d(channels_in,channels_out,kernel_size,stride=s,padding=padding,bias=bias)#la partie réelle du vecteur de conv
		self.convImagLayer = torch.nn.ConvTranspose3d(channels_in,channels_out,kernel_size,stride=s,padding=padding,bias=bias)#la partie imaginaire du vecteur de conv
