"""Frozen seed42 six-method complexity audit; no training, no result selection.

One fold01 test input per dataset; native time-domain inputs for time baselines.
FLOPs via fvcore with explicit 2-FLOP MAC/custom handlers; timings include CPU
ordered decoding and transfers but exclude TF preprocessing/checkpoint loading.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,csv,time,hashlib,gc,warnings,argparse,platform
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=['/tmp/rem_dual_head_formal',str(ROOT),'/tmp/rem_complexity_tools']
import numpy as np
import torch,h5py
from torch import nn
from torch.nn import functional as F
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.flop_count import _DEFAULT_SUPPORTED_OPS
from fvcore.nn.jit_handles import get_shape
from src.models.baselines.dual_endpoint import DualEndpointBaseline
from experiments.phase_resampling_20261027.architecture import AnchorConditionedREM
from experiments.phase_resampling_20261027.frozen_bridge import structured_decode
OUT=ROOT/'docs/paper/sections/complexity_20261030'
NAMES=('mmecg_tcn','radarode_mtl','cross_modal_conformer','wavegru_net','attention_dcgan','rem_net')
LABELS=dict(zip(NAMES,('Chen-TCN','radarODE-MTL','Conformer','WaveGRU-Net','Attention-DCGAN','REM-Net')))
BACKBONES=dict(zip(NAMES,('TCN (time)','ResNet (TF)','Conformer','CNN--BiGRU','Attn. CNN','TF stem + TCN')))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def size(v):
 s=get_shape(v)
 return int(np.prod(s)) if s is not None else 0

def doubled(fn,operator):
 def handle(i,o):
  value=fn(i,o)
  result=Counter({k:2*v for k,v in value.items()}) if isinstance(value,Counter) else Counter({operator.split('::')[-1]:2*value})
  # Stock fvcore MAC handlers omit fused bias additions; add them explicitly.
  bias=(i[2] if operator in {'aten::_convolution','aten::linear'} else i[0] if operator=='aten::addmm' else None)
  if bias is not None and get_shape(bias) is not None:result['bias_add']=size(o[0])
  return result
 return handle

def per_output(factor):return lambda i,o:factor*size(o[0])
def interpolation(i,o):
 # 1D:2mul+1add;2D:4mul+3add. Coordinate weights counted separately.
 dims=len(get_shape(o[0]));points=int(np.prod(get_shape(o[0])[2:]))
 return (3 if dims==3 else 7)*size(o[0])+points*(3 if dims==3 else 6)

def grid(i,o):return 7*size(o[0])+10*size(i[1])//2

def gru(i,o):
 # aten::gru output [B,T,D*H] for batch_first; layers from argument4.
 x=get_shape(i[0]);y=get_shape(o[0]);layers=i[4].toIValue();directions=2 if i[7].toIValue() else 1
 batchfirst=i[8].toIValue();b,t=(x[0],x[1]) if batchfirst else (x[1],x[0]);h=y[-1]//directions
 inp=x[-1];total=0
 for layer in range(layers):
  # 3 input/recurrent gate matrices; two biases and reset/update/state elementwise.
  total+=b*t*directions*(6*h*(inp+h)+6*h+23*h);inp=directions*h
 return total

def deform(i,o):
 # torchvision input,weight,offset,mask,bias,...
 x,w,y=get_shape(i[0]),get_shape(i[1]),get_shape(o[0]);b,cout,ho,wo=y;ci=x[1];kh,kw=w[2:]
 mac=b*ho*wo*cout*w[1]*kh*kw
 sampled=b*ho*wo*ci*kh*kw
 coords=b*ho*wo*kh*kw
 return Counter(deform_conv_mac=2*mac,deform_bilinear_mask=8*sampled,deform_coordinates=8*coords)

def attention(i,o):
 q,k,v=map(get_shape,i[:3]);n=int(np.prod(q[:-2]));l,d=q[-2:];s=k[-2];dv=v[-1]
 return 2*n*l*s*(d+dv)+3*n*l*s

def native_attention(i,o):
 b,t,e=get_shape(i[0]);heads=i[4].toIValue()
 return Counter(attention_projection=8*b*t*e*e+4*b*t*e,
                attention_products=4*b*t*t*e,attention_softmax_scale=3*b*heads*t*t+b*t*e)


def handlers():
 h={k:doubled(v,k) for k,v in _DEFAULT_SUPPORTED_OPS.items() if k in {'aten::addmm','aten::bmm','aten::_convolution','aten::einsum','aten::matmul','aten::mm','aten::linear'}}
 h.update({'aten::_native_multi_head_attention':native_attention,'aten::grid_sampler':grid,'aten::gru':gru,'torchvision::deform_conv2d':deform,
  'aten::upsample_linear1d':interpolation,'aten::upsample_bilinear2d':interpolation,
  'aten::scaled_dot_product_attention':attention,'aten::_scaled_dot_product_flash_attention_for_cpu':attention,
  'aten::adaptive_avg_pool1d':lambda i,o:size(i[0]),'aten::avg_pool1d':lambda i,o:size(o[0])*i[1].toIValue()[0],
  'aten::softmax':per_output(3),'aten::_softmax':per_output(3),'aten::gelu':per_output(8),
  'aten::sigmoid':per_output(4),'aten::silu':per_output(5),'aten::tanh':per_output(5),
  'aten::glu':per_output(5),'aten::max_pool2d':per_output(0),'aten::max_pool1d':per_output(0),
  'aten::linspace':lambda i,o:2*size(o[0]),'aten::feature_dropout':per_output(0),
  'aten::numel':per_output(0),'aten::pad':per_output(0),'aten::meshgrid':per_output(0),'aten::repeat':per_output(0),
  'aten::mean':lambda i,o:size(i[0]),'aten::sum':lambda i,o:max(0,size(i[0])-size(o[0])),
  'aten::std':lambda i,o:4*size(i[0]),'aten::var':lambda i,o:4*size(i[0])})
 for name in ('add','sub','mul','div','pow','sqrt','exp','log','neg','abs','clamp','clamp_min','maximum','minimum','relu','leaky_relu'):
  h['aten::'+name]=per_output(1)
 for name in ('add_','mul_','div_','sub_','rsub','leaky_relu_'):h['aten::'+name]=per_output(1)
 return h

class Forward(nn.Module):
 def __init__(self,m,name,ds):super().__init__();self.model=m;self.name=name;self.ds=ds
 def forward(self,x):
  if self.name=='rem_net':z=self.model(x)
  else:
   if self.name=='radarode_mtl' and self.ds=='crrvs':x=F.interpolate(x,size=(71,240),mode='bilinear',align_corners=False)
   z=self.model(wsst=x) if self.name=='radarode_mtl' else self.model(rcg=x)
  return z['shape'],z['anchor_logits'].sigmoid()

class Inference:
 def __init__(self,wrapped):self.wrapped=wrapped
 def __call__(self,x):
  shape,prob=self.wrapped(x)
  # REM phase_pd already uses same ordered decoder internally for phase bounds;
  # decode full event output too, exactly like evaluator (not discarded at deployment).
  anchors=structured_decode(prob.detach().cpu().numpy()[0])
  return shape,anchors

def checkpoint(ds,name):
 if name=='rem_net':
  selected=ROOT/f'experiments/phase_resampling_20261027/formal_ablation_20261027/runs/{ds}/fold_01/A4/selected.json'
  return Path(json.loads(selected.read_text())['checkpoint'])
 return ROOT/f'experiments_mmecg/q1_dual_endpoint_baselines_{"mmecg" if ds=="mmecg" else "crrvs"}/seed_42/{name}/fold_01/checkpoints/best.pt'

def inputs(ds):
 base=Path('/home/qhh2237/Datasets')/('MMECG/processed/loso_v2' if ds=='mmecg' else 'CR_RVS/processed/q1_continuous_kfold')/'fold_01'
 with h5py.File(base/'test.h5') as h:
  rcg=np.asarray(h['rcg'][0:1],np.float32);rid=h['record_id'][0];rid=rid.decode() if isinstance(rid,bytes) else str(rid)
  start=int(h['start_idx'][0]);sid=int(h['subject_id'][0])
 if ds=='mmecg':
  path=Path('/home/qhh2237/Datasets/MMECG/processed/wsst8s_subject_pool')/f'subject_{sid:02d}.h5'
  with h5py.File(path) as h:
   ids=[x.decode() if isinstance(x,bytes) else str(x) for x in h['record_id'][:]]
   index=next(i for i,(r,s) in enumerate(zip(ids,h['start_idx'][:])) if r==rid and s==start)
   tf=np.asarray(h['wsst'][index:index+1],np.float32)
 else:
  path=base/'test_wsst8s.h5'
  with h5py.File(path) as h:tf=np.asarray(h['wsst'][0:1],np.float32)
 assert tf.shape==(1,50,71,240) if ds=='mmecg' else tf.shape==(1,1,65,93)
 return torch.from_numpy(tf),torch.from_numpy(rcg),dict(record_id=rid,start=start,subject=sid,tf_path=str(path),native_path=str(base/'test.h5'),tf_sha256=hashlib.sha256(tf.tobytes()).hexdigest(),rcg_sha256=hashlib.sha256(rcg.tobytes()).hexdigest())

@torch.no_grad()
def timing(fn,x,device,warmup=50,repetitions=200):
 def sync():
  if device=='cuda':torch.cuda.synchronize()
 for _ in range(warmup):fn(x)
 sync();times=[]
 for _ in range(repetitions):
  sync();start=time.perf_counter();fn(x);sync();times.append((time.perf_counter()-start)*1000)
 return dict(median_ms=float(np.median(times)),p25_ms=float(np.quantile(times,.25)),p75_ms=float(np.quantile(times,.75)),samples_ms=times)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile-only',action='store_true');ap.add_argument('--export-only',action='store_true');args=ap.parse_args()
 if args.export_only:export(json.loads((OUT/'measurements.json').read_text()));return
 OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 results=[]
 for ds in ('mmecg','crrvs'):
  tf,rcg,source=inputs(ds)
  for name in NAMES:
   path=checkpoint(ds,name);saved=torch.load(path,map_location='cpu',weights_only=False)
   model=AnchorConditionedREM('phase_pd',in_bins=50 if ds=='mmecg' else 1) if name=='rem_net' else DualEndpointBaseline(name,in_bins=50 if ds=='mmecg' else 1)
   model.load_state_dict(saved['model'],strict=True);model.eval()
   assert all(p.requires_grad for p in model.parameters())
   params=sum(p.numel() for p in model.parameters() if p.requires_grad)
   wrapped=Forward(model,name,ds).eval();x=tf if name in ('rem_net','radarode_mtl') else rcg
   with torch.no_grad(),warnings.catch_warnings():
    warnings.simplefilter('ignore')
    analysis=FlopCountAnalysis(wrapped,(x,)).set_op_handle(**handlers()).tracer_warnings('none').uncalled_modules_warnings(False).unsupported_ops_warnings(False)
    flops=analysis.total();unsupported=dict(analysis.unsupported_ops());byop=dict(analysis.by_operator())
   result=dict(dataset='MMECG' if ds=='mmecg' else 'CR_RVS',method=LABELS[name],name=name,backbone=BACKBONES[name],params=params,params_m=params/1e6,flops=flops,flops_g=flops/1e9,input_shape=list(x.shape),input_source=source,checkpoint=str(path),checkpoint_sha256=sha(path),checkpoint_epoch=saved.get('epoch'),unsupported_ops=unsupported,flops_by_operator=byop)
   print('PROFILE',ds,name,params,flops/1e9,'unsupported',unsupported,flush=True)
   if unsupported:raise RuntimeError('Unaccounted operators: '+str(unsupported))
   if not args.profile_only:
    fn=Inference(wrapped.to('cuda'));gpu=timing(fn,x.to('cuda'),'cuda')
    wrapped.cpu();fn=Inference(wrapped);cpu=timing(fn,x,'cpu')
    result.update(gpu_latency=gpu,cpu_latency=cpu,gpu_model=torch.cuda.get_device_name(0))
    print('TIMING',ds,name,'GPU',gpu['median_ms'],'CPU',cpu['median_ms'],flush=True)
   results.append(result)
   (OUT/('profile_audit.json' if args.profile_only else 'measurements.json')).write_text(json.dumps(results,indent=2))
   del wrapped,model,saved;gc.collect();torch.cuda.empty_cache()
 if not args.profile_only:export(results)

def export(results):
 import fvcore
 columns=['method','dataset','Params_M','FLOPs_G','GPU_latency_ms','CPU_latency_ms','GPU_model']
 with (OUT/'complexity.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
  for r in results:w.writerow(dict(zip(columns,(r['method'],r['dataset'],r['params_m'],r['flops_g'],r['gpu_latency']['median_ms'],r['cpu_latency']['median_ms'],r['gpu_model']))))
 for ds,file in (('MMECG','table_complexity.tex'),('CR_RVS','table_complexity_crrvs.tex')):
  lines=[r'\begin{table}[t]',r'\centering',r'\caption{Model complexity and inference cost ('+ds.replace('_',r'\_')+r' input, one 8-s window).}',r'\label{tab:complexity-'+ds.lower()+r'}',r'\setlength{\tabcolsep}{2.3pt}',r'\scriptsize',r'\resizebox{\columnwidth}{!}{%',r'\begin{tabular}{@{}llrrr@{}}',r'\toprule',r'Method & Backbone & Params & FLOPs & Latency \\',r' & & (M) & (G) & (ms) \\',r'\midrule']
  for r in results:
   if r['dataset']!=ds:continue
   if r['name']=='rem_net':lines.extend([r'\midrule',r'\rowcolor{gray!12}'])
   label=r'\textbf{REM-Net (ours)}' if r['name']=='rem_net' else r['method']+r'$^{\dagger}$'
   lines.append(f"{label} & {r['backbone']} & {r['params_m']:.2f} & {r['flops_g']:.2f} & {r['gpu_latency']['median_ms']:.2f}"+r' \\')
  lines.extend([r'\bottomrule',r'\end{tabular}%',r'}',r'\vspace{2pt}',r'\parbox{\columnwidth}{\scriptsize $^{\dagger}$Adapted backbones with common shape/event output format. Time: native 1-D radar; TF: time--frequency input. Parameters include all trainable allocated layers. FLOPs use two operations per multiply--add, with custom operator estimates, excluding TF preprocessing. Latency: median of 200 FP32 batch-1 runs after 50 warm-ups on an NVIDIA GeForce RTX 4080 SUPER, including ordered decoding and REM-Net phase resampling.}',r'\end{table}'])
  (OUT/file).write_text('\n'.join(lines)+'\n')
 write_report(results)
 (OUT/'environment.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,fvcore=fvcore.__version__,gpu=torch.cuda.get_device_name(0),cpu=platform.processor(),threads=torch.get_num_threads(),tf32=False,warmup=50,repetitions=200,script_sha256=sha(__file__)),indent=2))



def write_report(results):
 out=OUT;r=results
 lines=['# Seed-42 trained-model complexity audit','',
 'Inference only: six trained methods, fold01 selected checkpoint per dataset. No training, evaluation-result changes, checkpoint selection, or paper-table replacement. REM-Net is full phase_pd (A4); baselines are the five original adapted dual-head checkpoints. Seed42 is the experiment/base-seed label; phase runs use base42+fold internally. Exact checkpoint paths, epochs and SHA256 are in measurements.json.',
 '', '## Inputs and timing scope',
 'Exactly one fixed real first test window per dataset, batch1. No GT ECG enters forward/decoding. MMECG TF input [1,50,71,240]; CR_RVS log-STFT [1,1,65,93]. The four time-domain methods receive native cardiac RCG [1,50,1600] / [1,1,1600], NOT WSST/STFT. radarODE-MTL consumes TF; CR_RVS resizes 65x93 to71x240 with bilinear align_corners=False within the measured wrapper. Feature extraction/filtering, disk I/O, checkpoint loading, and initial host-to-GPU input transfer are excluded. Required input adaptation/interpolation is included.',
 'GPU NVIDIA GeForce RTX4080SUPER, FP32, no AMP, TF32 disabled for matmul and cuDNN. All models model.eval() + torch.no_grad(); cudnn.benchmark=False. 50 warmup runs then200 timed runs EACH on GPU and CPU. Synchronize CUDA immediately before starting and after finishing EACH measured run; time.perf_counter measures elapsed wall time. Median reported; all200 raw times and IQR retained. No other compute workload on GPU during profiling. GPU clocks are not pinned; these are this machine/runtime measurements, not device-independent costs.',
 'End-to-end latency includes sigmoid, device-to-host event-probability transfer, and the frozen NumPy/SciPy ordered Q/R/S/T decoder. REM-Net phase_pd additionally includes its internal ordered decoding, soft-argmax refinement, phase-grid construction, two grid_sample calls, and full-window context interpolation. A final full-event decoding is run for REM-Net as for each baseline (same as evaluator), rather than reusing undocumented internal intermediates. One shape per 8s window is timed; this is NOT dense per-beat record assembly latency.',
 'CPU: AMD Ryzen9 7900X3D12-Core Processor, torch intra-op/inter-op threads1; OMP/MKL/OpenBLAS threads1. CPU timings include the same full inference/decode pipeline and are supplementary only.',
 '', '## Parameters',
 'Exact sum(p.numel() for p in model.parameters() if p.requires_grad), prior to changing any requires_grad flags. Shape and event readouts included; buffers excluded. The loaded baseline implementations retain unused original core readout/task layers as trainable allocated parameters (especially radarODE-MTL). They are included as requested; forward FLOPs/latency count only executed operations. These are adapted-model allocation counts, not the parameter sizes of the original published models, nor active-backbone-only counts. No unused layer was deleted for timing.',
 '', '## FLOPs convention / missing-operator audit',
 'All12 cells use fvcore FlopCountAnalysis with the same custom handlers. https://detectron2.readthedocs.io/en/latest/modules/fvcore.html documents that stock fvcore counts FMA as1. This audit replaces conv/linear/matrix/einsum handlers with2x stock MAC counts; normalization and scalar arithmetic are not globally doubled. Fused convolution/linear bias additions omitted by stock MAC handlers are explicitly added once per output element. G=1e9. Values are analytical operation estimates, not hardware instruction counts; activations/transcendentals follow explicit equivalent-scalar-cost rules in script.',
 'Custom supplements: fused native MHA includes Q/K/V/output projections8*B*T*E^2, bias4*B*T*E, score/value products4*B*T^2*E, softmax3*B*heads*T^2 and scalingB*T*E. SDPA includes both attention matrix products. Bidirectional GRU counts both directions/all three gates: per step/direction6H(I+H)+6H+23H (2-operation MACs, biases, reset/update/nonlinear estimates). Linear interpolation3/output, bilinear7/output, plus coordinate-weight estimates. grid_sample is NOT omitted: stock default4/output is replaced with7/output (4mult+3add) plus10/grid point for coordinates/weights. This also counts degenerate height1 sampling conservatively as bilinear; it is not asserted to equal the optimized kernel instruction count. REM-Net uses two sampling calls.',
 'radarODE actual backbone uses the official pure-PyTorch dcnv2.DeformConv2d (nets/backbone/dcnv2.py), NOT the separate torchvision DeformableConv2d in nets/units.py. Its interpolation/gather-weight arithmetic is visible to fvcore through add/sub/rsub/mul and normal convolutions; no duplicate manual deform-convolution lump is added. A torchvision custom handler is registered for completeness but is not exercised (confirmed per-operator audit).',
 'Zero-floating-arithmetic/index/control operations: padding, meshgrid, repeat, gather/view/reshape, numel, max-pool comparisons and eval dropout. Counted scalar elementwise arithmetic, GELU8/output, sigmoid4/output, SiLU5/output, tanh5/output, softmax3/output; equivalent-operation conventions stated rather than treated as exact transcendental instruction cost. NumPy/SciPy discrete peak search/argmax/order checks and CPU/Python control are outside the tensor FLOP trace; their wall-clock cost is INCLUDED in latency. Data-dependent REM pair/soft-argmax trace is specialized to the fixed input; no universal exact per-window FLOPs claim is made.',
 'unsupported_ops is empty in every final tensor trace (measurements.json). Manually ignored indexing/comparison/control operations above are explicitly distinguished from arithmetic omissions. Native attention, GRU, grid sampling and upsampling are therefore not silently missing.',
 '', '## Results (GPU table; CPU for reference)',
 '| Dataset | Method | Params (M) | FLOPs (G) | GPU median (ms) | CPU single-thread median (ms) |',
 '|---|---|---:|---:|---:|---:|']
 for x in r:lines.append(f"| {x['dataset']} | {x['method']} | {x['params_m']:.6f} | {x['flops_g']:.6f} | {x['gpu_latency']['median_ms']:.3f} | {x['cpu_latency']['median_ms']:.3f} |")
 lines+=['','## Reproduce / outputs',
 '`/home/qhh2237/miniconda3/envs/cyberbrain/bin/python -B scripts/profile_model_complexity.py`',
 'Local fvcore and lightweight dependencies installed under /tmp/rem_complexity_tools only, without changing trained-model dependencies: `python -m pip install --target /tmp/rem_complexity_tools fvcore iopath yacs termcolor portalocker tabulate --no-deps` (existing NumPy/PyYAML/tqdm reused). Versions recorded in environment.json.',
 'complexity.csv: all12 measurements. table_complexity.tex: MMECG single-column main table. table_complexity_crrvs.tex: CR_RVS single-column supplement. measurements.json: checkpoint provenance, input fingerprint, per-operator FLOPs, unsupported audit, all400 timing samples/model-dataset. environment.json: software/device/settings/script hash. profile_audit.json contains the final FLOP-only audit; measurements.json combines those counts with the measured timings.',
 'LaTeX requires booktabs, graphicx and xcolor with table option. The display table contains GPU latency only. No performance/highest-efficiency superiority claim is inferred from this audit. Existing paper tables/figures are untouched.']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print('Report generated',len(r),'rows')

if __name__=='__main__':main()
