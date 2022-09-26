#!/usr/bin/python

import sys, time, os, pdb, argparse, pickle, subprocess, glob, cv2
import numpy as np
from shutil import rmtree

# import scenedetect
# from scenedetect.video_manager import VideoManager
# from scenedetect.scene_manager import SceneManager
# from scenedetect.frame_timecode import FrameTimecode
# from scenedetect.stats_manager import StatsManager
# from scenedetect.detectors import ContentDetector

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

# from detectors import S3FD



# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========

def bb_intersection_over_union(boxA, boxB):
  
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
 
  interArea = max(0, xB - xA) * max(0, yB - yA)
 
  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
 
  iou = interArea / float(boxAArea + boxBArea - interArea)
 
  return iou

# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========

def track_shot(opt,scenefaces):

  iouThres  = 0.08     # Minimum IOU between consecutive face detections
  tracks    = []

  while True:
    track     = []
    for framefaces in scenefaces:
      for face in framefaces:
        if track == []:
          track.append(face)
          framefaces.remove(face)
        elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
          iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
          if iou > iouThres:
            track.append(face)
            framefaces.remove(face)
            continue
        else:
          break

    if track == []:
      break
    elif len(track) > opt.min_track:
      
      framenum    = np.array([ f['frame'] for f in track ])
      bboxes      = np.array([np.array(f['bbox']) for f in track])

      framenum[0]=0
      framenum[-1]=len(scenefaces)-1
      frame_i   = np.arange(framenum[0],framenum[-1]+1)
      # frame_i   = np.arange(0,len(scenefaces))


      bboxes_i    = []
      for ij in range(0,4):
        interpfn  = interp1d(framenum, bboxes[:,ij])
        bboxes_i.append(interpfn(frame_i))
      bboxes_i  = np.stack(bboxes_i, axis=1)

      if max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1])) > opt.min_face_size:
        tracks.append({'frame':frame_i,'bbox':bboxes_i})

  return tracks

# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========
        
def crop_video(opt,track,cropfile):

  flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.png'))
  flist.sort()

  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, opt.frame_rate, (224,224))

  dets = {'x':[], 'y':[], 's':[]}

  for det in track['bbox']:

    dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
    dets['y'].append((det[1]+det[3])/2) # crop center x 
    dets['x'].append((det[0]+det[2])/2) # crop center y

  # Smooth detections
  dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
  dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
  dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

  for fidx, frame in enumerate(track['frame']):

    cs  = opt.crop_scale

    bs  = dets['s'][fidx]   # Detection box size
    bsi = int(bs*(1+2*cs))  # Pad videos by this amount 

    image = cv2.imread(flist[frame])
    
    frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
    my  = dets['y'][fidx]+bsi  # BBox center Y
    mx  = dets['x'][fidx]+bsi  # BBox center X

    face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
    
    vOut.write(cv2.resize(face,(224,224)))

  audiotmp    = os.path.join(opt.tmp_dir,opt.reference,'audio.wav')
  audiostart  = (track['frame'][0])/opt.frame_rate
  audioend    = (track['frame'][-1]+1)/opt.frame_rate

  vOut.release()

  # ========== CROP AUDIO FILE ==========

  command = ("scp %s %s" % (os.path.join(opt.avi_dir,opt.reference,'audio.wav'),audiotmp)) 
  output = subprocess.call(command, shell=True, stdout=None)

  # if output != 0:
  #   pdb.set_trace()

  sample_rate, audio = wavfile.read(audiotmp)

  # ========== COMBINE AUDIO AND VIDEO FILES ==========

  command = ("ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile,audiotmp,cropfile))
  output = subprocess.call(command, shell=True, stdout=None)

  # if output != 0:
  #   pdb.set_trace()

  print('Written %s'%cropfile)

  os.remove(cropfile+'t.avi')

  print('Mean pos: x %.2f y %.2f s %.2f'%(np.mean(dets['x']),np.mean(dets['y']),np.mean(dets['s'])))

  return {'track':track, 'proc_track':dets}

# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========

# def inference_video(opt):

#   DET = S3FD(device='cuda')

#   flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.png'))
#   flist.sort()

#   dets = []
      
#   for fidx, fname in enumerate(flist):

#     start_time = time.time()
    
#     image = cv2.imread(fname)

#     image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_np[250:,:] = 0
#     # image_np[:,360:,:] = 0
#     # print(image_np.shape)
#     # cv2.imwrite('/home/panzexu/test.png', image_np)
#     # assert 1==2
#     bboxes = DET.detect_faces(image_np, conf_th=0.15, scales=[opt.facedet_scale])
#     print(bboxes)
#     assert 1==2

#     dets.append([]);
#     for bbox in bboxes:
#       dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})

#     elapsed_time = time.time() - start_time

#     print('%s-%05d; %d dets; %.2f Hz' % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),fidx,len(dets[-1]),(1/elapsed_time))) 

#   return dets
    
from mtcnn.mtcnn import MTCNN
def inference_video(opt):

  DET = MTCNN()


  flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.png'))
  flist.sort()

  dets = []
      
  for fidx, fname in enumerate(flist):

    start_time = time.time()
    
    image = cv2.imread(fname)

    # image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_np[250:,:] = 0
    # image_np[:,360:,:] = 0
    # print(image_np.shape)
    # cv2.imwrite('/home/panzexu/test.png', image_np)
    # assert 1==2

    bboxes = DET.detect_faces(image)

    dets.append([]);
    for bbox in bboxes:
      bbox['box'][2] += bbox['box'][0]
      bbox['box'][3] += bbox['box'][1]
      dets[-1].append({'frame':fidx, 'bbox':(bbox['box']), 'conf':bbox['confidence']})

    elapsed_time = time.time() - start_time

    print('%s-%05d; %d dets; %.2f Hz' % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),fidx,len(dets[-1]),(1/elapsed_time))) 

  return dets


# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========
def main(opt):
  setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
  setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
  setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
  setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))
  setattr(opt,'frames_dir',os.path.join(opt.data_dir,'pyframes'))

  # ========== DELETE EXISTING DIRECTORIES ==========

  if os.path.exists(os.path.join(opt.crop_dir,opt.reference)):
    rmtree(os.path.join(opt.crop_dir,opt.reference))

  if os.path.exists(os.path.join(opt.avi_dir,opt.reference)):
    rmtree(os.path.join(opt.avi_dir,opt.reference))

  if os.path.exists(os.path.join(opt.frames_dir,opt.reference)):
    rmtree(os.path.join(opt.frames_dir,opt.reference))

  if os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
    rmtree(os.path.join(opt.tmp_dir,opt.reference))

  # ========== MAKE NEW DIRECTORIES ==========

  os.makedirs(os.path.join(opt.crop_dir,opt.reference))
  os.makedirs(os.path.join(opt.avi_dir,opt.reference))
  os.makedirs(os.path.join(opt.frames_dir,opt.reference))
  os.makedirs(os.path.join(opt.tmp_dir,opt.reference))

  # ========== CONVERT VIDEO AND EXTRACT FRAMES ==========

  command = ("scp %s %s" % (opt.videofile,os.path.join(opt.avi_dir,opt.reference,'video.avi')))
  output = subprocess.call(command, shell=True, stdout=None)

  # command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),os.path.join(opt.frames_dir,opt.reference,'%06d.jpg'))) 
  # output = subprocess.call(command, shell=True, stdout=None)
  command = ("ffmpeg -i %s -vf fps=25 %s" % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),os.path.join(opt.frames_dir,opt.reference,'%06d.png'))) 
  output = subprocess.call(command, shell=True, stdout=None)

  command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),os.path.join(opt.avi_dir,opt.reference,'audio.wav'))) 
  output = subprocess.call(command, shell=True, stdout=None)


  # ========== FACE DETECTION ==========

  faces = inference_video(opt)
  print(len(faces))

  # ========== FACE TRACKING ==========

  alltracks = []

  alltracks.extend(track_shot(opt,faces))

  # ========== FACE TRACK CROP ==========
  if opt.reference[5] == 'F':
    gender=['M','F','U_1','U_2','U_3','U_4','U_5']
  else:
    gender=['F','M','U_1','U_2','U_3','U_4','U_5']

  for ii, track in enumerate(alltracks):
    crop_video(opt,track,os.path.join(opt.crop_dir,opt.reference,opt.reference+'_%s'%gender[ii]))

  rmtree(os.path.join(opt.frames_dir,opt.reference))
  rmtree(os.path.join(opt.tmp_dir,opt.reference))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = "FaceTracker");
  parser.add_argument('--data_dir',       type=str, default='/home/panzexu/datasets/iemocap/tmp/', help='Output direcotry');
  parser.add_argument('--videofile',      type=str, default='/home/panzexu/datasets/iemocap/video/Ses02F_impro01.avi',   help='Input video file');
  parser.add_argument('--reference',      type=str, default='demo',   help='Video reference');
  parser.add_argument('--facedet_scale',  type=float, default=0.25, help='Scale factor for face detection');
  parser.add_argument('--crop_scale',     type=float, default=0.40, help='Scale bounding box');
  parser.add_argument('--min_track',      type=int, default=500,  help='Minimum facetrack duration');
  parser.add_argument('--frame_rate',     type=int, default=25,   help='Frame rate');
  parser.add_argument('--num_failed_det', type=int, default=3000,   help='Number of missed detections allowed before tracking is stopped');
  parser.add_argument('--min_face_size',  type=int, default=5,  help='Minimum face size in pixels');
  opt = parser.parse_args();

  # opt.reference=opt.videofile.split('/')[-1][:-4]
  # main(opt)
  # assert 1==7


  # # convert bframes
  dataset='/home/panzexu/datasets/iemocap/'
  orig_path = dataset + 'orig/'
  video_path = dataset + 'video/'
  # if not os.path.exists(video_path):
  #   os.makedirs(video_path)
  # for path, dirs ,files in os.walk(orig_path):
  #   for filename in files:
  #     if filename[-4:] =='.avi' and filename[0]!='.':
  #       if filename[:5] =='Ses02':
  #         command ="ffmpeg -i %s -codec copy -bsf:v mpeg4_unpack_bframes %s;" % (path + filename,video_path+filename)
  #         os.system(command)

  vid_list = []
  for path, dirs ,files in os.walk(video_path):
    for filename in files:
      if filename[-4:] =='.avi' and filename[0]!='.':        
        if filename[:5] =='Ses03':
          opt.videofile=path+filename
          opt.reference=opt.videofile.split('/')[-1][:-4]
          vid_list.append(opt)
          # opt.videofile=path+filename
          # opt.reference=opt.videofile.split('/')[-1][:-4]
          main(opt)
          # assert 1==2

  # print(len(vid_list))
  # with Pool(22) as p:
  #   p.map(main, vid_list)
