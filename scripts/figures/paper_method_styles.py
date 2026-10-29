"""Canonical method encodings for paper figures; never encode fiducials here.

Existing published/archived figure files are not rewritten by importing this.
"""
METHODS=('rem_net','mmecg_tcn','radarode_mtl','cross_modal_conformer','wavegru_net','attention_dcgan')
METHOD_STYLES={
    'rem_net':dict(label='REM-Net',color='#1F4E79',linestyle='-',linewidth=1.4,zorder=20),
    'mmecg_tcn':dict(label='Chen-TCN',color='#D55E00',linestyle=(0,(4,2)),linewidth=.9,zorder=2),
    'radarode_mtl':dict(label='radarODE-MTL',color='#009E73',linestyle='-.',linewidth=.9,zorder=3),
    'cross_modal_conformer':dict(label='Conformer',color='#CC79A7',linestyle=(0,(2,1)),linewidth=.9,zorder=4),
    'wavegru_net':dict(label='WaveGRU-Net',color='#E69F00',linestyle=':',linewidth=.9,zorder=5),
    'attention_dcgan':dict(label='Attention-DCGAN',color='#56B4E9',linestyle=(0,(6,2)),linewidth=.9,zorder=6),
}
