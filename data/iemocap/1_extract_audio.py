import os
import numpy as np 
import argparse
import scipy.io.wavfile as wavfile

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

np.random.seed(0)

def extract_wav_from_mp4(line):
	# Extract .wav file from mp4
	fn = line[2]
	video_from_path=args.video_data_direc +line[0]+'/'+line[1]+'/'+line[2]+'.avi'
	audio_save_path=args.audio_data_direc +line[0]+'/'+line[1]+'/'+line[2]+'.wav'

	if not os.path.exists(audio_save_path.rsplit('/', 1)[0]):
		os.makedirs(audio_save_path.rsplit('/', 1)[0])

	if not os.path.exists(audio_save_path):
		os.system("ffmpeg -i %s %s"%(video_from_path, audio_save_path))

	sr, audio = wavfile.read(audio_save_path)
	assert sr==16000 , "sampling_rate mismatch"

	# mute the non-target's speech
	text_path = args.text_data_direc + line[0]+'/'+line[1]+'.txt'
	text_lst =open(text_path).read().splitlines()
	text_lst=list(filter(lambda x: x.startswith(fn), text_lst))

	audio_tgt = np.zeros(audio.shape)
	for seg in text_lst:
		time = seg.split(' ')[1]
		start_time = round(float(time.split('-')[0][1:])*16000)-320
		end_time = round(float(time.split('-')[1][:-2])*16000)-320
		audio_tgt[start_time:end_time] = audio[start_time:end_time]

	audio_tgt = np.divide(audio_tgt, np.max(np.abs(audio_tgt)))
	write_wav(audio_save_path , audio_tgt)

	sample_length = audio.shape[0]
	return sample_length # In seconds


def main(args):
	# Get test set list of audios
	for path, dirs ,files in os.walk(args.video_data_direc):
		for filename in files:
			if filename[-4:] =='.avi' and filename[0] != '.':
				ln = [path.split('/')[-2],path.split('/')[-1], filename.split('.')[0]]
				sample_length = extract_wav_from_mp4(ln)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='iemocap dataset')
	parser.add_argument('--video_data_direc', default='/home/panzexu/datasets/iemocap/processed/face_crop/', type=str)
	parser.add_argument('--audio_data_direc', default='/home/panzexu/datasets/iemocap/processed/audio_clean/' ,type=str)
	parser.add_argument('--text_data_direc', default='/home/panzexu/datasets/iemocap/processed/transcriptions/' ,type=str)
	args = parser.parse_args()
	
	main(args)