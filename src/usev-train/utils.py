import numpy as np
import math
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.utils.data as data
import scipy.io.wavfile as wavfile
from itertools import permutations
from apex import amp
import tqdm
import os

EPS = 1e-8
MAX_INT16 = np.iinfo(np.int16).max

def read_wav(fname, normalize=True):
    """
    Read wave files using scipy.io.wavfile(support multi-channel)
    """
    # samps_int16: N x C or N
    #   N: number of samples
    #   C: number of channels
    sampling_rate, samps_int16 = wavfile.read(fname)
    # N x C => C x N
    samps = samps_int16.astype(np.float)
    # tranpose because I used to put channel axis first
    if samps.ndim != 1:
        samps = np.transpose(samps)
    # normalize like MATLAB and librosa
    if normalize:
        samps = samps / MAX_INT16
    return sampling_rate, samps

class dataset(data.Dataset):
    def __init__(self,
                mix_lst_path,
                audio_direc,
                visual_direc,
                mixture_direc,
                batch_size,
                partition='test',
                sampling_rate=16000,
                max_length=6,
                mix_no=2):

        self.minibatch =[]
        self.audio_direc = audio_direc
        self.visual_direc = visual_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.max_length = max_length
        self.batch_size=batch_size

        mix_lst=open(mix_lst_path).read().splitlines()
        mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))

        sorted_mix_lst = sorted(mix_lst, key=lambda data: float(data.split(',')[-1]), reverse=True)

        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end

    def __getitem__(self, index):
        batch_lst = self.minibatch[index]
        min_length = int(float(batch_lst[-1].split(',')[-1])*self.sampling_rate)

        mixtures=[]
        audios=[]
        visuals=[]
        labels_tgt=[]
        labels_int=[]
        for line in batch_lst:
            # read mixtures
            mixture_path=self.mixture_direc+self.partition+'/'+ line.replace(',','_').replace('/','_')[:200]+'.wav'
            if not os.path.exists(mixture_path):
                mixture_path = mixture_path.replace('/2_mix_sparse','/3_mix_sparse')
            
            _, mixture = read_wav(mixture_path)
            mixture = mixture[:min_length]
            mixtures.append(mixture)


            line=line.split(',')
            c=0
            # read visual images
            visual_path=self.visual_direc+ self.partition+'/'+ line[c*6+1]+'/'+line[c*6+2]+'/'+line[c*6+3]+'.npy'
            visual = np.load(visual_path)
            start = int(line[c*6+4])
            end = int(line[c*6+5])
            visual = visual[round(start/16000*25):round(end/16000*25)]

            length = math.floor(min_length/self.sampling_rate*25)
            visual = visual[:length,...]

            if visual.shape[0] < length:
                visual = np.pad(visual, ((0,int(length - visual.shape[0])),(0,0)), mode = 'edge')
            visuals.append(visual)
            

            audio_path=self.audio_direc+ self.partition+'/'+line[c*6+1]+'/'+line[c*6+2]+'/'+line[c*6+3]+'.wav'
            _, audio = read_wav(audio_path)
            audio = audio[start:end]
            audio = audio[:min_length]
            audios.append(audio)

            # read label of speaker activity binary mask
            tgt_time = line[c*6+3].split('_')[-4:]
            tgt_time = list(map(int, tgt_time))
            label_tgt = np.zeros(tgt_time[1] - tgt_time[0])
            label_tgt[tgt_time[2] - tgt_time[0]: tgt_time[3] - tgt_time[0]] = 1
            label_tgt = label_tgt[start:end]
            label_tgt = label_tgt[:min_length]
            labels_tgt.append(label_tgt)

            c_int = 1
            int_start = int(line[c_int*6+4])
            int_end = int(line[c_int*6+5])
            int_time = line[c_int*6+3].split('_')[-4:]
            int_time = list(map(int, int_time))
            label_int = np.zeros(int_time[1] - int_time[0])
            label_int[int_time[2] - int_time[0]: int_time[3] - int_time[0]] = 1
            label_int = label_int[int_start:int_end]
            label_int = label_int[:min_length]
            labels_int.append(label_int)

            if len(line) > 20:
                c_int = 2
                int_start = int(line[c_int*6+4])
                int_end = int(line[c_int*6+5])
                int_time = line[c_int*6+3].split('_')[-4:]
                int_time = list(map(int, int_time))
                label_int_2 = np.zeros(int_time[1] - int_time[0])
                label_int_2[int_time[2] - int_time[0]: int_time[3] - int_time[0]] = 1
                label_int_2 = label_int_2[int_start:int_end]
                label_int_2 = label_int_2[:min_length]

                label_int = label_int + label_int_2
                label_int = np.clip(label_int, 0,1)


        a_mix = np.asarray(mixtures)[...,:self.max_length*self.sampling_rate]
        a_tgt = np.asarray(audios)[...,:self.max_length*self.sampling_rate]
        v_tgt = np.asarray(visuals)[:,:self.max_length*25,...]
        labels_tgt = np.asarray(labels_tgt)[...,:self.max_length*self.sampling_rate]
        labels_int = np.asarray(labels_int)[...,:self.max_length*self.sampling_rate]

        return a_mix, a_tgt, v_tgt, labels_tgt, labels_int

    def __len__(self):
        return len(self.minibatch)


class DistributedSampler(data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()
            ind = torch.randperm(int(len(self.dataset)/self.num_replicas), generator=g)*self.num_replicas
            indices = []
            for i in range(self.num_replicas):
                indices = indices + (ind+i).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = indices[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_dataloader(args, partition):
    datasets = dataset(
                mix_lst_path=args.mix_lst_path,
                audio_direc=args.audio_direc,
                visual_direc=args.visual_direc,
                mixture_direc=args.mixture_direc,
                batch_size=args.batch_size,
                max_length=args.max_length,
                partition=partition,
                mix_no=args.C)

    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank) if args.distributed else None

    generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = (sampler is None),
            num_workers = args.num_workers,
            sampler=sampler)

    return sampler, generator

@amp.float_function
def cal_SISNR(source, estimate_source):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """
    assert source.size() == estimate_source.size()

    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)

    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)

    return sisnr


def cal_SDR(source, estimate_source):
    assert source.size() == estimate_source.size()
    
    # estimate_source += EPS # the estimated source is zero sometimes

    noise = source - estimate_source
    ratio = torch.sum(source ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sdr = 10 * torch.log10(ratio + EPS)

    return sdr


def cal_logEnergy(source):
    ratio = torch.sum(source ** 2, axis = -1)
    sdr = 10 * torch.log10(ratio + EPS)
    return sdr

