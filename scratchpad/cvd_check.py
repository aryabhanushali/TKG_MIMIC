import itertools, sys, numpy as np
M={'protan':[[0.152286,1.052583,-0.204868],[0.114503,0.786281,0.099216],[-0.003882,-0.048116,1.051998]],
   'deutan':[[0.367322,0.860646,-0.227968],[0.280085,0.672501,0.047413],[-0.011820,0.042940,0.968881]],
   'tritan':[[1.255528,-0.076749,-0.178779],[-0.078411,0.930809,0.147602],[0.004733,0.691367,0.303900]]}
def lin(h):
    c=np.array([int(h[i:i+2],16)/255 for i in (1,3,5)]); return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def oklab(l):
    A=np.array([[0.4122214708,0.5363325363,0.0514459929],[0.2119034982,0.6806995451,0.1073969566],[0.0883024619,0.2817188376,0.6299787005]])
    B=np.array([[0.2104542553,0.7936177850,-0.0040720468],[1.9779984951,-2.4285922050,0.4505937099],[0.0259040371,0.7827717662,-0.8086757660]])
    return B@np.cbrt(A@np.clip(l,0,1))
def lum(l): return 0.2126*l[0]+0.7152*l[1]+0.0722*l[2]
cols=sys.argv[1].split(','); surf=sys.argv[2] if len(sys.argv)>2 else '#ffffff'
for a,b in itertools.combinations(cols,2):
    la,lb=lin(a),lin(b); out={'normal':np.linalg.norm(oklab(la)-oklab(lb))*100}
    for k,m in M.items(): out[k]=np.linalg.norm(oklab(np.array(m)@la)-oklab(np.array(m)@lb))*100
    print(a,b,' '.join(f'{k}={v:5.1f}' for k,v in out.items()), 'worstCVD=%.1f'%min(out['protan'],out['deutan'],out['tritan']))
ls=lum(lin(surf))
for c in cols:
    lc=lum(lin(c)); print(c,'contrast vs surface %.2f'%((max(lc,ls)+.05)/(min(lc,ls)+.05)))
