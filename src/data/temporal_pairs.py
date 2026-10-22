"""Sequence-safe temporal positive-pair dataset and distinct-sequence batches."""
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler

from src.data.video_loader import R01VideoClipDataset


class TemporalPositivePairDataset(Dataset):
    def __init__(self,records,transform,clip_length=16,stride=4,max_offset=3,augment=False,seed=42):
        self.video=R01VideoClipDataset(records,transform,clip_length,stride); self.max_offset=int(max_offset); self.augment=bool(augment); self.seed=int(seed); self.epoch=0
        self.by_sequence=defaultdict(list)
        for i,c in enumerate(self.video.clips): self.by_sequence[c['sequence']].append(i)
        self.positions={}
        for seq,ids in self.by_sequence.items():
            for pos,idx in enumerate(ids): self.positions[idx]=(seq,pos)

    @property
    def clips(self): return self.video.clips
    def set_epoch(self,epoch): self.epoch=int(epoch)
    def __len__(self): return len(self.video)
    def __getitem__(self,index):
        seq,pos=self.positions[index]; ids=self.by_sequence[seq]
        rng=random.Random(self.seed+self.epoch*1_000_003+index)
        possible=[p for p in range(max(0,pos-self.max_offset),min(len(ids),pos+self.max_offset+1)) if 0<abs(p-pos)<=self.max_offset]
        if not possible: raise ValueError(f'No temporal positive exists for {seq} clip position {pos}')
        positive_pos=rng.choice(possible); xa,_=self.video[index]; xb,_=self.video[ids[positive_pos]]
        if self.augment:
            xa=self._jitter(xa,rng); xb=self._jitter(xb,rng)
        return xa,xb,seq,ids[positive_pos]

    @staticmethod
    def _jitter(x,rng):
        # Mild per-clip brightness/contrast changes in normalized input space.
        contrast=rng.uniform(.95,1.05); brightness=rng.uniform(-.03,.03)
        return x*contrast+brightness


class DistinctSequenceBatchSampler(Sampler):
    """Each batch draws at most one anchor from a sequence, ensuring other rows are negatives."""
    def __init__(self,dataset,batch_size,shuffle=True,seed=42): self.dataset=dataset; self.batch_size=int(batch_size); self.shuffle=shuffle; self.seed=int(seed); self.epoch=0
    def set_epoch(self,epoch): self.epoch=int(epoch)
    def __iter__(self):
        rng=random.Random(self.seed+self.epoch); groups={s:list(v) for s,v in self.dataset.by_sequence.items()}; seqs=list(groups)
        for ids in groups.values(): rng.shuffle(ids)
        if self.shuffle: rng.shuffle(seqs)
        pointers={s:0 for s in seqs}; batches=[]; active=[s for s in seqs if groups[s]]
        while active:
            if self.shuffle: rng.shuffle(active)
            chosen=active[:self.batch_size]; batch=[]
            for s in chosen:
                batch.append(groups[s][pointers[s]]); pointers[s]+=1
            batches.append(batch); active=[s for s in active if pointers[s]<len(groups[s])]
        yield from batches
    def __len__(self):
        # Conservative upper bound; batch sampler may have partial batches.
        return max(len(self.dataset.by_sequence), (len(self.dataset)+self.batch_size-1)//self.batch_size)
