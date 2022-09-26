import argparse
import torch
from utils import *
import os
from avDprnn import avDprnn
from mir_eval.separation import bss_eval_sources
from pystoi import stoi
from pypesq import pesq
import csv

EPS = 1e-8
MAX_INT16 = np.iinfo(np.int16).max

def cal_logpower(source):
    ratio = torch.sum(source ** 2, axis = -1)
    sdr = 10 * torch.log10(ratio/source.shape[-1]*16000 + EPS)
    return sdr

    
def write_wav(fname, samps, sampling_rate=16000, normalize=True):
    """
    Write wav files in int16, support single/multi-channel
    """
    # for multi-channel, accept ndarray [Nsamples, Nchannels]
    samps = np.divide(samps, np.max(np.abs(samps)))

    # same as MATLAB and kaldi
    if normalize:
        samps = samps * MAX_INT16
        samps = samps.astype(np.int16)
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile instead
    wavfile.write(fname, sampling_rate, samps)

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
                batch_size=1,
                partition='test',
                sampling_rate=16000,
                mix_no=2):

        self.minibatch =[]
        self.audio_direc = audio_direc
        self.visual_direc = visual_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C=mix_no

        mix_csv=open(mix_lst_path).read().splitlines()
        self.mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_csv))

        # self.mix_lst = self.mix_lst[:100]

        
    def __getitem__(self, index):
        line = self.mix_lst[index]
        line_cache = line

        mixture_path=self.mixture_direc+self.partition+'/'+ line.replace(',','_').replace('/','_')[:200]+'.wav'
        if not os.path.exists(mixture_path):
            mixture_path = mixture_path.replace('/2_mix_sparse','/3_mix_sparse')
        _, mixture = read_wav(mixture_path)
        
        min_length = mixture.shape[0]

        c=0
        line=line.split(',')
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

        
        # read audio
        audio_path=self.audio_direc+ self.partition+'/'+line[c*6+1]+'/'+line[c*6+2]+'/'+line[c*6+3]+'.wav'
        _, audio = read_wav(audio_path)
        audio = audio[start:end]
        audio = audio[:min_length]

        # read label of speaker activity binary mask
        tgt_time = line[c*6+3].split('_')[-4:]
        tgt_time = list(map(int, tgt_time))
        label_tgt = np.zeros(tgt_time[1] - tgt_time[0])
        label_tgt[tgt_time[2] - tgt_time[0]: tgt_time[3] - tgt_time[0]] = 1
        label_tgt = label_tgt[start:end]

        # read label of speaker activity binary mask
        c = 1
        int_start = int(line[c*6+4])
        int_end = int(line[c*6+5])
        int_time = line[c*6+3].split('_')[-4:]
        int_time = list(map(int, int_time))
        label_int = np.zeros(int_time[1] - int_time[0])
        label_int[int_time[2] - int_time[0]: int_time[3] - int_time[0]] = 1
        label_int = label_int[int_start:int_end]

        if len(line) > 20:
            c = 2
            int_start = int(line[c*6+4])
            int_end = int(line[c*6+5])
            int_time = line[c*6+3].split('_')[-4:]
            int_time = list(map(int, int_time))
            label_int_2 = np.zeros(int_time[1] - int_time[0])
            label_int_2[int_time[2] - int_time[0]: int_time[3] - int_time[0]] = 1
            label_int_2 = label_int_2[int_start:int_end]

            label_int = label_int + label_int_2
            label_int = np.clip(label_int, 0,1)

        return mixture, audio, visual, label_tgt, label_int, line, line_cache

    def __len__(self):
        return len(self.mix_lst)

    def _audio_norm(self,audio):
        return np.divide(audio, np.max(np.abs(audio)))


def segment_utt(a_mix, a_tgt, a_est, label_tgt, label_int):
    utt_list = []
    for utt in [a_mix, a_tgt, a_est]:
        utt_1 = utt[((label_tgt == label_int) & (label_tgt == 0))]
        utt_3 = utt[((label_tgt == label_int) & (label_tgt == 1))]
        utt_2 = utt[((label_tgt != label_int) & (label_tgt == 1))]
        utt_4 = utt[((label_tgt != label_int) & (label_tgt == 0))]
        assert utt.shape[-1] == (utt_1.shape[-1]+ utt_2.shape[-1]+ utt_3.shape[-1]+ utt_4.shape[-1])
        utt_list.append([utt_1,utt_2,utt_3,utt_4]) 
    return utt_list[0], utt_list[1], utt_list[2]

def eval_segment_utt(a_mix, a_tgt,a_est):
    a_mix_1, a_mix_2, a_mix_3, a_mix_4= a_mix[0], a_mix[1], a_mix[2], a_mix[3]
    a_tgt_1, a_tgt_2, a_tgt_3, a_tgt_4= a_tgt[0], a_tgt[1], a_tgt[2], a_tgt[3]
    a_est_1, a_est_2, a_est_3, a_est_4= a_est[0], a_est[1], a_est[2], a_est[3]

    energy_1, sisnr_2 , sdr_2, energy_2, sisnr_3,sdr_3, energy_3, energy_4 = None, None, None, None, None, None, None, None


    if a_mix_1.shape[-1]!=0:
        energy_1 = cal_logpower(a_est_1)

    if a_mix_2.shape[-1]!=0:
        energy_2 = cal_logpower(a_est_2)
        sisnr_2 = cal_SISNR(a_tgt_2, a_est_2)
        sdr_2 = cal_SDR(a_tgt_2, a_est_2)

    if a_mix_3.shape[-1]!=0:
        energy_3 = cal_logpower(a_est_3)
        sisnr_3 = cal_SISNR(a_tgt_3, a_est_3)
        sdr_3 = cal_SDR(a_tgt_3, a_est_3)

    if a_mix_4.shape[-1]!=0:
        energy_4 = cal_logpower(a_est_4)

    return energy_1, sisnr_2 , sdr_2, energy_2, sisnr_3,sdr_3, energy_3, energy_4

def eval_segment_weighted_sisdr(a_mix_u, a_tgt_u,a_est_u, label_tgt, label_int):
    utt_list = []
    for utt in [a_mix_u, a_tgt_u,a_est_u]:
        utt_1 = utt[((label_tgt == label_int) & (label_tgt == 0))]
        utt_3 = utt[((label_tgt == label_int) & (label_tgt == 1))]
        utt_2 = utt[((label_tgt != label_int) & (label_tgt == 1))]
        utt_4 = utt[((label_tgt != label_int) & (label_tgt == 0))]
        assert utt.shape[-1] == (utt_1.shape[-1]+ utt_2.shape[-1]+ utt_3.shape[-1]+ utt_4.shape[-1])
        utt_list.append([utt_1,utt_2,utt_3,utt_4]) 

    a_mix, a_tgt,a_est = utt_list[0], utt_list[1], utt_list[2]
    a_mix_1, a_mix_2, a_mix_3, a_mix_4= a_mix[0], a_mix[1], a_mix[2], a_mix[3]
    a_tgt_1, a_tgt_2, a_tgt_3, a_tgt_4= a_tgt[0], a_tgt[1], a_tgt[2], a_tgt[3]
    a_est_1, a_est_2, a_est_3, a_est_4= a_est[0], a_est[1], a_est[2], a_est[3]

    sisnr_1, sisnr_2, sisnr_3, sisnr_4, sisnr_5, sisnr_6, sisnr_7 = None, None, None, None, None, None, None

    avg_sisnri = cal_SISNR(a_tgt_u, a_est_u)
    non_ca_energy = cal_logpower(a_est_u)
    
    if a_mix_3.shape[-1]==0:
        if a_mix_2.shape[-1]==0:
            sisnr_1 = cal_logpower(a_est_u)
            non_ca_energy = None
            pass
        elif a_mix_2.shape[-1]!=0:
            sisnr_2 = avg_sisnri
    else:
        a = a_mix_3.shape[-1]
        b = a_mix_2.shape[-1]+a_mix_3.shape[-1]+a_mix_4.shape[-1]
        ratio = a/b
        if ratio <=0.2:
            sisnr_3 = avg_sisnri
        elif ratio <=0.4:
            sisnr_4 = avg_sisnri
        elif ratio <=0.6:
            sisnr_5 = avg_sisnri
        elif ratio <=0.8:
            sisnr_6 = avg_sisnri
        elif ratio <=1.0:
            sisnr_7 = avg_sisnri

    return non_ca_energy, sisnr_1, sisnr_2, sisnr_3, sisnr_4, sisnr_5, sisnr_6, sisnr_7

def main(args):
    # Model
    model = avDprnn(args.N, args.L, args.B, args.H, args.K, args.R, args.C)
    # model = avConvtasnet()

    model = model.cuda()

    pretrained_model = torch.load('%smodel_dict_best.pt' % args.continue_from, map_location='cpu')['model']
    state = model.state_dict()
    for key in state.keys():
        pretrain_key = 'module.' + key
        # if pretrain_key in pretrained_model.keys():
        state[key] = pretrained_model[pretrain_key]
    model.load_state_dict(state)

    datasets = dataset(
                mix_lst_path=args.mix_lst_path,
                audio_direc=args.audio_direc,
                visual_direc=args.visual_direc,
                mixture_direc=args.mixture_direc,
                mix_no=args.C)

    test_generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = False,
            num_workers = args.num_workers)

    model.eval()
    with torch.no_grad():
        avg_energy_1 = []
        avg_sisnri_2 = []
        avg_sisnri_3 = []
        avg_energy_4 = []
        avg_sdr_2  = []
        avg_sdr_3  = []
        avg_energy_2  = []
        avg_energy_3 = []

        avg_u_sisnr_0 = []
        avg_u_sisnr_1 = []
        avg_u_sisnr_2 = []
        avg_u_sisnr_3 = []
        avg_u_sisnr_4 = []
        avg_u_sisnr_5 = []
        avg_u_sisnr_6 = []
        avg_u_sisnr_7 = []


        for i, (a_mix, a_tgt, v_tgt, label_tgt,label_int, flist, line_cache) in enumerate(tqdm.tqdm(test_generator)):
            a_mix = a_mix.cuda().squeeze().float().unsqueeze(0)
            a_tgt = a_tgt.cuda().squeeze().float().unsqueeze(0)
            v_tgt = v_tgt.cuda().squeeze().float().unsqueeze(0)
            label_tgt = label_tgt.cuda().long()
            label_int = label_int.cuda().long()

            a_est = model(a_mix, v_tgt)

            # evaluate weighted unified sisnr
            u_sisnr_0, u_sisnr_1, u_sisnr_2, u_sisnr_3, u_sisnr_4, u_sisnr_5, u_sisnr_6, u_sisnr_7  = eval_segment_weighted_sisdr(a_mix, a_tgt, a_est, label_tgt, label_int)
            
            if u_sisnr_0!=None:
                avg_u_sisnr_0.append(u_sisnr_0)
            if u_sisnr_1!=None:
                avg_u_sisnr_1.append(u_sisnr_1)
            if u_sisnr_2!=None:
                avg_u_sisnr_2.append(u_sisnr_2)
            if u_sisnr_3!=None:
                avg_u_sisnr_3.append(u_sisnr_3)
            if u_sisnr_4!=None:
                avg_u_sisnr_4.append(u_sisnr_4)
            if u_sisnr_5!=None:
                avg_u_sisnr_5.append(u_sisnr_5)
            if u_sisnr_6!=None:
                avg_u_sisnr_6.append(u_sisnr_6)
            if u_sisnr_7!=None:
                avg_u_sisnr_7.append(u_sisnr_7)

            # segmented loss
            a_mix, a_tgt, a_est = segment_utt(a_mix, a_tgt, a_est, label_tgt, label_int)
            energy_1, sisnr_2 , sdr_2, energy_2, sisnr_3,sdr_3, energy_3, energy_4  = eval_segment_utt(a_mix, a_tgt,a_est)

            if energy_1!=None:
                avg_energy_1.append(energy_1)
            if sisnr_2!=None:
                avg_sisnri_2.append(sisnr_2)
            if sisnr_3!=None:
                avg_sisnri_3.append(sisnr_3)
            if energy_4!=None:
                avg_energy_4.append(energy_4)
            if energy_2!=None:
                avg_energy_2.append(energy_2)
            if energy_3!=None:
                avg_energy_3.append(energy_3)
            if sdr_2!=None:
                avg_sdr_2.append(sdr_2)
            if sdr_3!=None:
                avg_sdr_3.append(sdr_3)


        avg_energy_1 = sum(avg_energy_1)/len(avg_energy_1)
        avg_sisnri_2 = sum(avg_sisnri_2)/len(avg_sisnri_2)
        avg_sisnri_3 = sum(avg_sisnri_3)/len(avg_sisnri_3)
        avg_energy_4 = sum(avg_energy_4)/len(avg_energy_4)
        avg_energy_2 = sum(avg_energy_2)/len(avg_energy_2)
        avg_energy_3 = sum(avg_energy_3)/len(avg_energy_3)
        avg_sdr_2 = sum(avg_sdr_2)/len(avg_sdr_2)
        avg_sdr_3 = sum(avg_sdr_3)/len(avg_sdr_3)
        print(avg_energy_1)
        print(avg_energy_2)
        print(avg_energy_3)
        print(avg_energy_4)
        print(avg_sisnri_2)
        print(avg_sisnri_3)
        print(avg_sdr_2)
        print(avg_sdr_3)
        print('*')

        avg_u_sisnr_0 = sum(avg_u_sisnr_0)/len(avg_u_sisnr_0)
        avg_u_sisnr_1 = sum(avg_u_sisnr_1)/len(avg_u_sisnr_1)
        avg_u_sisnr_2 = sum(avg_u_sisnr_2)/len(avg_u_sisnr_2)
        avg_u_sisnr_3 = sum(avg_u_sisnr_3)/len(avg_u_sisnr_3)
        avg_u_sisnr_4 = sum(avg_u_sisnr_4)/len(avg_u_sisnr_4)
        avg_u_sisnr_5 = sum(avg_u_sisnr_5)/len(avg_u_sisnr_5)
        avg_u_sisnr_6 = sum(avg_u_sisnr_6)/len(avg_u_sisnr_6)
        avg_u_sisnr_7 = sum(avg_u_sisnr_7)/len(avg_u_sisnr_7)
        print(avg_u_sisnr_0)
        print(avg_u_sisnr_1)
        print(avg_u_sisnr_2)
        print(avg_u_sisnr_3)
        print(avg_u_sisnr_4)
        print(avg_u_sisnr_5)
        print(avg_u_sisnr_6)
        print(avg_u_sisnr_7)
        print((avg_u_sisnr_2+avg_u_sisnr_3+avg_u_sisnr_4+avg_u_sisnr_5+avg_u_sisnr_6+avg_u_sisnr_7)/6)
        print('*')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("avConv-tasnet")
    
    # Dataloader
    parser.add_argument('--mix_lst_path', type=str, default='/home/panzexu/datasets/iemocap/uss/audio_mixture/2_mix_sparse_noise/mixture_data_list_mix.csv',
                        help='directory including train data')
    parser.add_argument('--audio_direc', type=str, default='/home/panzexu/datasets/iemocap/uss/audio_clean/',
                        help='directory including validation data')
    parser.add_argument('--visual_direc', type=str, default='/home/panzexu/datasets/iemocap/uss/visual_embedding/lip/',
                        help='directory including test data')
    parser.add_argument('--mixture_direc', type=str, default='/home/panzexu/datasets/iemocap/uss/audio_mixture/2_mix_sparse_noise/',
                        help='directory of audio')

    # Log and Visulization
    # parser.add_argument('--continue_from', type=str, 
    #     default='/home/panzexu/workspace/uss/src/av-dprnn/logs/avDprnn_06-02-2022(09:11:45)/')
    parser.add_argument('--continue_from', type=str, 
        default='/home/panzexu/workspace/uss/log/se/avDprnn_10-02-2022(23:58:14)/')

    # Training    
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Number of workers to generate minibatch')

    # Model hyperparameters
    parser.add_argument('--L', default=40, type=int,
                        help='Length of the filters in samples (80=5ms at 16kHZ)')
    parser.add_argument('--N', default=256, type=int,
                        help='Number of filters in autoencoder')
    parser.add_argument('--B', default=64, type=int,
                        help='Number of output channels')
    parser.add_argument('--C', type=int, default=2,
                        help='number of speakers to mix')
    parser.add_argument('--H', default=128, type=int,
                        help='Number of hidden size in rnn')
    parser.add_argument('--K', default=100, type=int,
                        help='Number of chunk size')
    parser.add_argument('--R', default=6, type=int,
                        help='Number of layers')


    args = parser.parse_args()

    main(args)
