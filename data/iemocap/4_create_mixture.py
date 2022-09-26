import os
import numpy as np 
import argparse
import scipy.io.wavfile as wavfile
import csv
import tqdm

np.random.seed(0)

MAX_INT16 = np.iinfo(np.int16).max

def write_wav(fname, samps, sampling_rate=16000, normalize=True):
	"""
	Write wav files in int16, support single/multi-channel
	"""
	# for multi-channel, accept ndarray [Nsamples, Nchannels]
	if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
		samps = np.transpose(samps)
		samps = np.squeeze(samps)
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


def create_mixture(data_list, noist_list,  data, length, w_file):
	for _ in tqdm.trange(length):
		mixtures=[data]
		cache = []

		# target speaker
		idx = np.random.randint(0, len(data_list))
		ID =  data_list[idx][0] + data_list[idx][2].split('_')[-5]
		cache.append(ID)

		time = data_list[idx][2].split('_')[-4:]
		time = list(map(int, time))

		tgt_tot_length = time[1] - time[0]

		tgt_length = np.random.randint(min(tgt_tot_length, 48000), min(tgt_tot_length +1, 160000))
		tgt_start = np.random.randint(0, tgt_tot_length - tgt_length+1)
		tgt_end = tgt_start + tgt_length

		# ratio = float("{:.2f}".format(np.random.uniform(-args.mix_db,args.mix_db)))
		ratio = 0

		mixtures = mixtures + list(data_list[idx]) + [tgt_start,tgt_end,ratio]


		# inteference speaker
		while len(cache) < (args.C):
			idx = np.random.randint(0, len(data_list))
			ID =  data_list[idx][0] + data_list[idx][2].split('_')[-5]
			if ID in cache:
				continue
		
			time = data_list[idx][2].split('_')[-4:]
			time = list(map(int, time))
			itf_tot_length = time[1] - time[0]
			if itf_tot_length < tgt_length:
				continue

			cache.append(ID)

			itf_start = np.random.randint(0, itf_tot_length - tgt_length+1)
			itf_end = itf_start + tgt_length

			ratio = float("{:.2f}".format(np.random.uniform(-args.mix_db,args.mix_db)))

			mixtures = mixtures + list(data_list[idx])  + [itf_start,itf_end,ratio]

		while True:
			noise_idx = np.random.randint(0, len(noist_list))
			noist_line = noist_list[noise_idx]
			_, audio_noise=read_wav(args.noise_data_direc+noist_line[0]+'/'+noist_line[1]+'/'+noist_line[2])
			noise_length = np.shape(audio_noise)[1]
			if noise_length < tgt_length:
				continue
			else:
				ratio = float("{:.2f}".format(np.random.uniform(-args.mix_db,args.mix_db)))
				mixtures = mixtures + list(noist_line) + [ratio]
				break

		mixtures.append(tgt_length/16000)
		w_file.writerow(mixtures)


def main(args):
	##############################
	##############################
	# Get test set list of audios
	test_utts = []
	for path, dirs ,files in os.walk(args.audio_data_direc+'test/'):
		for filename in files:
			if filename[-4:] =='.wav' and filename[0] != '.':
				ln = [path.split('/')[-2],path.split('/')[-1], filename.split('.')[0]]
				test_utts.append(ln)

	# Get val set list of audios
	val_utts = []
	for path, dirs ,files in os.walk(args.audio_data_direc+'val/'):
		for filename in files:
			if filename[-4:] =='.wav' and filename[0] != '.':
				ln = [path.split('/')[-2],path.split('/')[-1], filename.split('.')[0]]
				val_utts.append(ln)

	# Get train set list of audios
	train_utts = []
	for path, dirs ,files in os.walk(args.audio_data_direc+'train/'):
		for filename in files:
			if filename[-4:] =='.wav' and filename[0] != '.':
				ln = [path.split('/')[-2],path.split('/')[-1], filename.split('.')[0]]
				train_utts.append(ln)
	##############################
	##############################



	##############################
	##############################
	# Get test set list of noise
	test_noise = []
	for path, dirs ,files in os.walk(args.noise_data_direc+'test/'):
		for filename in files:
			if filename[-4:] =='.wav' and filename[0] != '.':
				ln = [path.split('/')[-2],path.split('/')[-1], filename]
				test_noise.append(ln)

	# Get val set list of noise
	val_noise = []
	for path, dirs ,files in os.walk(args.noise_data_direc+'val/'):
		for filename in files:
			if filename[-4:] =='.wav' and filename[0] != '.':
				ln = [path.split('/')[-2],path.split('/')[-1], filename]
				val_noise.append(ln)

	# Get train set list of noise
	train_noise = []
	for path, dirs ,files in os.walk(args.noise_data_direc+'train/'):
		for filename in files:
			if filename[-4:] =='.wav' and filename[0] != '.':
				ln = [path.split('/')[-2],path.split('/')[-1], filename]
				train_noise.append(ln)
	##############################
	##############################


	print("Creating mixture list")
	f_talk=open(args.mixture_data_list,'w')
	w_talk=csv.writer(f_talk)

	create_mixture(test_utts,test_noise, 'test', args.test_samples, w_talk)
	create_mixture(val_utts,val_noise, 'val', args.val_samples, w_talk)
	create_mixture(train_utts,train_noise, 'train', args.train_samples, w_talk)

	return


def create_audio(args):
	# create mixture
	mixture_data_list = open(args.mixture_data_list).read().splitlines()
	print(len(mixture_data_list))

	for line in tqdm.tqdm(mixture_data_list,desc = "Generating audio mixtures"):
		data = line.split(',')
		save_direc=args.mixture_audio_direc+data[0]+'/'
		if not os.path.exists(save_direc):
			os.makedirs(save_direc)
		
		mixture_save_path=save_direc+line.replace(',','_').replace('/','_')[:200] +'.wav'
		if os.path.exists(mixture_save_path):
			continue

		# read target audio
		c = 0
		_, audio_clean=read_wav(args.audio_data_direc+data[0]+'/'+data[c*6+1]+'/'+data[c*6+2]+'/'+data[c*6+3]+'.wav')
		start = int(data[4])
		end = int(data[5])
		audio_clean = audio_clean[start:end]

		target_power = np.linalg.norm(audio_clean, 2)**2 / audio_clean.size


		_, audio_noise=read_wav(args.noise_data_direc+'/'+data[(args.C)*6 +1]+'/'+data[(args.C)*6+2]+'/'+data[(args.C)*6+3],normalize=False)
		audio_noise = audio_noise[0,:(end-start)]
		scalar = (10**((float(data[(args.C)*6 +4])-5)/20))
		noise_power = np.linalg.norm(audio_noise, 2)**2 / audio_noise.size

		if target_power!=0:
			audio_noise = audio_noise * scalar * np.sqrt(target_power/noise_power)
		else:
			audio_noise = audio_noise * scalar

		audio_mix = audio_noise

		# read inteference audio
		for c in range(1, args.C):
			audio_path=args.audio_data_direc+data[0]+'/'+data[c*6+1]+'/'+data[c*6+2]+'/'+data[c*6+3]+'.wav'
			_, audio = read_wav(audio_path)
			start = int(data[c*6+4])
			end = int(data[c*6+5])
			audio = audio[start:end]
			scalar = (10**(float(data[c*6+6])/20))
			intef_power = np.linalg.norm(audio, 2)**2 / audio.size
			if target_power!=0 and intef_power !=0:
				audio = audio * scalar * np.sqrt(target_power/intef_power)
			else:
				audio = audio * scalar

			audio_mix = audio_mix + audio


		audio_save = audio_clean + audio_mix

		audio_save = np.divide(audio_save, np.max(np.abs(audio_save)))
		write_wav(mixture_save_path, audio_save)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='iemocap dataset')
	parser.add_argument('--audio_data_direc', default='/home/panzexu/datasets/iemocap/uss/audio_clean/' ,type=str)
	parser.add_argument('--noise_data_direc', default='/home/panzexu/datasets/wham_noise/' ,type=str)
	parser.add_argument('--mixture_audio_direc', default='/home/panzexu/datasets/iemocap/uss/audio_mixture/3_mix_sparse_noise/', type=str)
	parser.add_argument('--C', default = 2 ,type=int)
	parser.add_argument('--mix_db', default = 5, type=float)
	parser.add_argument('--train_samples', default = 200000, type=int)
	parser.add_argument('--val_samples', default = 5000, type=int)
	parser.add_argument('--test_samples', default = 3000, type=int)
	parser.add_argument('--mixture_data_list', default = 'mixture_data_list_2mix.csv', type=str)
	args = parser.parse_args()
	
	# create data list
	# main(args)

	# generate 2 speaker mixture
	create_audio(args)

	# generate 3 speaker mixture
	args.mixture_data_list = 'mixture_data_list_3mix.csv'
	args.C = 3
	create_audio(args)

	