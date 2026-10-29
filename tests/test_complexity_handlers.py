"""Analytic checks for FLOP rules absent/different in stock fvcore."""
import importlib.util
from pathlib import Path
import torch
from torch import nn
spec=importlib.util.spec_from_file_location('complexity_audit',Path(__file__).resolve().parents[1]/'scripts/profile_model_complexity.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def count(model,x):
 with torch.no_grad():
  a=c.FlopCountAnalysis(model.eval(),(x,)).set_op_handle(**c.handlers()).tracer_warnings('none').unsupported_ops_warnings(False).uncalled_modules_warnings(False)
  total=a.total();assert not a.unsupported_ops()
 return total

def test_convolution_counts_two_operations_per_mac():
 assert count(nn.Conv1d(2,3,3,bias=False),torch.zeros(1,2,10))==2*3*8*2*3

def test_bidirectional_gru_includes_both_directions_and_gates():
 class Model(nn.Module):
  def __init__(self):super().__init__();self.gru=nn.GRU(4,3,batch_first=True,bidirectional=True)
  def forward(self,x):return self.gru(x)[0]
 assert count(Model(),torch.zeros(1,5,4))==5*2*(6*3*(4+3)+6*3+23*3)

def test_grid_sampler_includes_interpolation_and_coordinates():
 class Model(nn.Module):
  def forward(self,x):
   grid=x.new_zeros(1,1,5,2)
   return torch.nn.functional.grid_sample(x,grid,align_corners=True,padding_mode='border')
 assert count(Model(),torch.zeros(1,2,1,10))==7*(1*2*1*5)+10*5

def test_fused_attention_includes_projections_and_attention_products():
 class Model(nn.Module):
  def __init__(self):super().__init__();self.attn=nn.MultiheadAttention(8,2,batch_first=True)
  def forward(self,x):return self.attn(x,x,x,need_weights=False)[0]
 # Native CPU eval fastpath: QKV+output, biases, QK/AV products, softmax/scaling.
 assert count(Model(),torch.zeros(1,4,8))==8*4*8*8+4*4*8+4*4*4*8+3*2*4*4+4*8
