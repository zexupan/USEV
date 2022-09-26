import os
import numpy as np 
import argparse
import scipy.io.wavfile as wavfile
import csv
import tqdm
import math
import cv2 as cv

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


def write_npy(fname, file):
	fdir = os.path.dirname(fname)
	if fdir and not os.path.exists(fdir):
		os.makedirs(fdir)
	np.save(fname,file)



def visual_images(visual_path):
	normMean = 0.52411
	normStd = 0.22226
	captureObj = cv.VideoCapture(visual_path)
	roiSequence = list()
	while (captureObj.isOpened()):
	    ret, frame = captureObj.read()
	    if ret == True:
	        grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	        grayed = grayed/255
	        roiSequence.append(grayed)
	    else:
	        break
	captureObj.release()
	img = np.stack(roiSequence, axis=0)
	img = (img - normMean) / normStd
	return img


def main(args):
	# Get test set list of audios
	talks = []
	for path, dirs ,files in os.walk(args.audio_data_direc):
		for filename in files:
			if filename[-4:] =='.wav' and filename[0] != '.':
				ln = [path.split('/')[-2],path.split('/')[-1], filename.split('.')[0]]
				talks.append(ln)
				# print(ln)

	np.random.shuffle(talks)
	train_talks = talks[:240]
	val_talks = talks[240:270]
	test_talks = talks[270:]

	train_utts = []
	for line in train_talks:
		text_path = args.text_data_direc + line[0]+'/'+line[1]+'.txt'
		text_lst =open(text_path).read().splitlines()
		text_lst=list(filter(lambda x: x.startswith(line[2]), text_lst))

		_, audio=read_wav(args.audio_data_direc+line[0]+'/'+line[1]+'/'+line[2]+'.wav')
		visual_path = args.video_data_direc+line[0]+'/'+line[1]+'/'+line[2]+'.avi'
		images = visual_images(visual_path)

		# cut from the random point of the silence region
		for i, (seg) in enumerate(text_lst):

			time = seg.split(' ')[1]
			start_time = round(float(time.split('-')[0][1:])*16000)-320
			end_time = round(float(time.split('-')[1][:-2])*16000)-320

			if end_time <= start_time: continue
			if audio.shape[0] < start_time: continue
			if audio.shape[0] < end_time: continue

			if i == 0:
				pre_end_time = start_time
			else:
				pre_end_time = nex_start_time

			assert pre_end_time <= start_time

			if i == (len(text_lst) -1):
				nex_start_time = end_time
			else:
				nex_time = text_lst[i+1].split(' ')[1]
				nex_start_time = round(float(nex_time.split('-')[0][1:])*16000)-320 
				if nex_start_time < end_time: continue
				if nex_start_time != end_time: nex_start_time = np.random.randint(end_time, nex_start_time)
			
			ln_append = line + [str(pre_end_time)] + [str(nex_start_time)] + [str(start_time)] + [str(end_time)]
			train_utts.append(ln_append)

			audio_save_path = args.clean_audio_data_direc + 'train/' + '/'.join(line[:-1]) + '/' + '_'.join(ln_append[2:]) + '.wav'
			audio_tgt = audio[pre_end_time:nex_start_time]
			# audio_tgt = np.divide(audio_tgt, np.max(np.abs(audio_tgt)))
			write_wav(audio_save_path , audio_tgt)

			v_pre_end_time = round(pre_end_time/16000*25)
			v_nex_start_time = round(nex_start_time/16000*25)
			visual_img = images[v_pre_end_time:v_nex_start_time]
			visual_save_path = args.visual_frame_direc + 'train/' + '/'.join(line[:-1]) + '/' + '_'.join(ln_append[2:]) + '.npy'
			write_npy(visual_save_path,visual_img)


	val_utts = []
	for line in val_talks:
		text_path = args.text_data_direc + line[0]+'/'+line[1]+'.txt'
		text_lst =open(text_path).read().splitlines()
		text_lst=list(filter(lambda x: x.startswith(line[2]), text_lst))

		_, audio=read_wav(args.audio_data_direc+line[0]+'/'+line[1]+'/'+line[2]+'.wav')
		visual_path = args.video_data_direc+line[0]+'/'+line[1]+'/'+line[2]+'.avi'
		images = visual_images(visual_path)

		# cut from the random point of the silence region
		for i, (seg) in enumerate(text_lst):

			time = seg.split(' ')[1]
			start_time = round(float(time.split('-')[0][1:])*16000)-320
			end_time = round(float(time.split('-')[1][:-2])*16000)-320

			if end_time <= start_time: continue
			if audio.shape[0] < start_time: continue
			if audio.shape[0] < end_time: continue

			if i == 0:
				pre_end_time = start_time
			else:
				pre_end_time = nex_start_time

			assert pre_end_time <= start_time

			if i == (len(text_lst) -1):
				nex_start_time = end_time
			else:
				nex_time = text_lst[i+1].split(' ')[1]
				nex_start_time = round(float(nex_time.split('-')[0][1:])*16000)-320 
				if nex_start_time < end_time: continue
				if nex_start_time != end_time: nex_start_time = np.random.randint(end_time, nex_start_time)
			
			ln_append = line + [str(pre_end_time)] + [str(nex_start_time)] + [str(start_time)] + [str(end_time)]
			val_utts.append(ln_append)

			audio_save_path = args.clean_audio_data_direc + 'val/' + '/'.join(line[:-1]) + '/' + '_'.join(ln_append[2:]) + '.wav'
			audio_tgt = audio[pre_end_time:nex_start_time]
			# audio_tgt = np.divide(audio_tgt, np.max(np.abs(audio_tgt)))
			write_wav(audio_save_path , audio_tgt)

			v_pre_end_time = round(pre_end_time/16000*25)
			v_nex_start_time = round(nex_start_time/16000*25)
			visual_img = images[v_pre_end_time:v_nex_start_time]
			visual_save_path = args.visual_frame_direc + 'val/' + '/'.join(line[:-1]) + '/' + '_'.join(ln_append[2:]) + '.npy'
			write_npy(visual_save_path,visual_img)


	test_utts = []
	for line in test_talks:
		text_path = args.text_data_direc + line[0]+'/'+line[1]+'.txt'
		text_lst =open(text_path).read().splitlines()
		text_lst=list(filter(lambda x: x.startswith(line[2]), text_lst))

		_, audio=read_wav(args.audio_data_direc+line[0]+'/'+line[1]+'/'+line[2]+'.wav')
		visual_path = args.video_data_direc+line[0]+'/'+line[1]+'/'+line[2]+'.avi'
		images = visual_images(visual_path)


		# cut from the random point of the silence region
		for i, (seg) in enumerate(text_lst):

			time = seg.split(' ')[1]
			start_time = round(float(time.split('-')[0][1:])*16000)-320
			end_time = round(float(time.split('-')[1][:-2])*16000)-320

			if end_time <= start_time: continue
			if audio.shape[0] < start_time: continue
			if audio.shape[0] < end_time: continue

			if i == 0:
				pre_end_time = start_time
			else:
				pre_end_time = nex_start_time

			assert pre_end_time <= start_time

			if i == (len(text_lst) -1):
				nex_start_time = end_time
			else:
				nex_time = text_lst[i+1].split(' ')[1]
				nex_start_time = round(float(nex_time.split('-')[0][1:])*16000)-320 
				if nex_start_time < end_time: continue
				if nex_start_time != end_time: nex_start_time = np.random.randint(end_time, nex_start_time)
			
			ln_append = line + [str(pre_end_time)] + [str(nex_start_time)] + [str(start_time)] + [str(end_time)]
			test_utts.append(ln_append)

			audio_save_path = args.clean_audio_data_direc + 'test/' + '/'.join(line[:-1]) + '/' + '_'.join(ln_append[2:]) + '.wav'
			audio_tgt = audio[pre_end_time:nex_start_time]
			# audio_tgt = np.divide(audio_tgt, np.max(np.abs(audio_tgt)))
			write_wav(audio_save_path , audio_tgt)

			v_pre_end_time = round(pre_end_time/16000*25)
			v_nex_start_time = round(nex_start_time/16000*25)
			visual_img = images[v_pre_end_time:v_nex_start_time]
			visual_save_path = args.visual_frame_direc + 'test/' + '/'.join(line[:-1]) + '/' + '_'.join(ln_append[2:]) + '.npy'
			write_npy(visual_save_path,visual_img)

	print(len(test_utts))
	print(len(val_utts))
	print(len(train_utts))

	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='iemocap dataset')
	parser.add_argument('--audio_data_direc', default='/home/panzexu/datasets/iemocap/processed/audio_clean/' ,type=str)
	parser.add_argument('--text_data_direc', default='/home/panzexu/datasets/iemocap/processed/transcriptions/' ,type=str)
	parser.add_argument('--clean_audio_data_direc', default='/home/panzexu/datasets/iemocap/uss/audio_clean/' ,type=str)
	parser.add_argument('--video_data_direc', default='/home/panzexu/datasets/iemocap/processed/face_crop/', type=str)
	parser.add_argument('--visual_frame_direc',default ='/home/panzexu/datasets/iemocap/uss/visual_embedding/image/', type=str)
	args = parser.parse_args()
	
	main(args)