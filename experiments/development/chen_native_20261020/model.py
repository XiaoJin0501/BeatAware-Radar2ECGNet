"""Paper-guided CNN/position-transformer + 512-RF causal categorical decoder.

Unspecified architectural details and dimensional contradictions are disclosed
in README. No official weights or official model implementation are claimed.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

BOS=256

def mulaw_encode(x):
    x=x.clamp(-1,1)
    comp=x.sign()*torch.log1p(255*x.abs())/math.log(256)
    return torch.round((comp+1)*127.5).long().clamp(0,255)

def mulaw_decode(code):
    comp=code.to(torch.float32)/127.5-1
    return comp.sign()*torch.expm1(comp.abs()*math.log(256))/255

class ConvPair(nn.Module):
    def __init__(self,a,b):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(a,b,7,padding=3),nn.ReLU(),nn.BatchNorm1d(b),
            nn.Conv1d(b,b,7,padding=3),nn.ReLU(),nn.BatchNorm1d(b))
    def forward(self,x):return self.net(x)

class Encoder(nn.Module):
    def __init__(self,bins=50):
        super().__init__();self.bins=bins
        widths=(1,8,16,32,32)
        self.temporal=nn.ModuleList([ConvPair(widths[i],widths[i+1]) for i in range(4)])
        self.position=nn.Linear(3,32)
        self.to_attention=nn.Linear(32,64)
        self.spatial=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,128,.1,batch_first=True),3,enable_nested_tensor=False)
        self.from_attention=nn.Linear(64,32)
        self.spatial_expand=nn.Linear(32,4)
        self.up=nn.ModuleList([nn.Sequential(nn.ConvTranspose1d(a,b,4,2,1),ConvPair(b,b)) for a,b in ((32,16),(16,8),(8,4))])
        self.fuse=nn.Conv1d(bins*4,4,1)
    def forward(self,radar,positions):
        if radar.ndim!=3 or radar.shape[1]!=self.bins or radar.shape[-1]%8:
            raise ValueError('Native radar [B,50,T], T divisible by8')
        b,r,t=radar.shape
        if positions.shape!=(b,r,3):raise ValueError('Measured3D cluster positions required')
        x=radar.reshape(b*r,1,t)
        for i,layer in enumerate(self.temporal):
            x=layer(x)
            if i<3:x=F.max_pool1d(x,2)
        descriptor=x.mean(-1).reshape(b,r,32)+self.position(positions)
        spatial=self.from_attention(self.spatial(self.to_attention(descriptor)))
        for layer in self.up:x=layer(x)
        assert x.shape[-1]==t
        x=x.reshape(b,r,4,t)*self.spatial_expand(spatial).unsqueeze(-1)
        return self.fuse(x.reshape(b,r*4,t))

class CausalBlock(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__();self.dilation=dilation
        self.conv=nn.Conv1d(channels,channels,2,dilation=dilation)
        self.drop=nn.Dropout(.1)
    def forward(self,x):
        h=self.conv(F.pad(x,(self.dilation,0)))
        return F.relu((x+self.drop(F.relu(h)))/math.sqrt(2))
    def step(self,x,buffer,index):
        old=buffer[:,:,index].clone();buffer[:,:,index]=x
        h=F.linear(old,self.conv.weight[:,:,0])+F.linear(x,self.conv.weight[:,:,1],self.conv.bias)
        out=F.relu((x+F.relu(h))/math.sqrt(2))
        return out,(index+1)%self.dilation

class Decoder(nn.Module):
    def __init__(self,channels=32):
        super().__init__();self.channels=channels
        self.embedding=nn.Embedding(257,channels,padding_idx=BOS)
        self.condition=nn.Conv1d(4,channels,1)
        self.blocks=nn.ModuleList([CausalBlock(channels,2**i) for i in range(9)])
        self.head=nn.Conv1d(channels,256,1)
        self.receptive_field=1+sum(2**i for i in range(9))
    def forward(self,previous_codes,condition):
        x=self.embedding(previous_codes).transpose(1,2)+self.condition(condition)
        for block in self.blocks:x=block(x)
        return self.head(x)
    def new_state(self,batch,device,dtype):
        return [[torch.zeros(batch,self.channels,b.dilation,device=device,dtype=dtype),0] for b in self.blocks]
    def step(self,previous_code,condition,state):
        if self.training:raise RuntimeError('Autoregressive cached inference requires eval')
        x=self.embedding(previous_code)+F.linear(condition,self.condition.weight[:,:,0],self.condition.bias)
        for block,entry in zip(self.blocks,state):
            x,entry[1]=block.step(x,entry[0],entry[1])
        return F.linear(x,self.head.weight[:,:,0],self.head.bias)

class ChenNativeAR(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Encoder();self.decoder=Decoder()
    def forward(self,radar,positions,codes):
        if radar.shape[-1]!=640 or codes.shape!=(len(radar),640):raise ValueError('Training uses640samples')
        previous=F.pad(codes[:,:-1],(1,0),value=BOS)
        return self.decoder(previous,self.encoder(radar,positions))
    @torch.inference_mode()
    def generate(self,radar,positions):
        if self.training:raise RuntimeError('Call eval before inference')
        condition=self.encoder(radar,positions)
        b,_,t=condition.shape
        state=self.decoder.new_state(b,condition.device,condition.dtype)
        previous=torch.full((b,),BOS,dtype=torch.long,device=radar.device)
        generated=[]
        for i in range(t):
            logits=self.decoder.step(previous,condition[:,:,i],state)
            previous=logits.argmax(-1)
            generated.append(previous)
        return torch.stack(generated,1)

def objective(logits,codes):
    if logits.shape[1:]!=(256,640):raise ValueError('256classes ×640timepoints')
    return F.cross_entropy(logits[:,:,512:],codes[:,512:])
