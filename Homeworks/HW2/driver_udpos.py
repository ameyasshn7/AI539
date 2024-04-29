import matplotlib
import torch
from torch.utils.data import DataLoader
from torchtext import datasets
from torch.utils.data.backward_compatibility import worker_init_fn
# from torch.utils.data.datapipes.utils.common import DILL_AVAILABLE



#Create data pipline

train_data = datasets.UDPOS(split='train')

#combine data elements from a batch

def pad_collate(batch):
    xx = [b[0] for b in batch]
    yy = [b[1] for b in batch]

    x_lens = [len(x) for x in xx]

    return xx, yy, x_lens

#making data loader

train_loader = DataLoader(dataset=train_data, batch_size=5,
                          shuffle = True, num_workers=1,
                          worker_init_fn=worker_init_fn,
                          drop_last=True, collate_fn=pad_collate)

#looking at the first batch

xx,yy,x_len = next(iter(train_loader))

#visualize PDS tagged sentence

def visualizeSentenceWithTags(text,udtags):
    print('Token' + ''.join([' '] * (15)) + 'PDS Tag')
    print('-----------------------------------------')
    for w,t in zip(text,udtags):
        print(w+''.join([' '] * (20-len(w))) + t)
        