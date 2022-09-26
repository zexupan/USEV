## USEV

A PyTorch implementation of the "USEV: Universal Speaker Extraction with Visual Cue"


## Project Structure

`/data`: Scripts to pre-process voxceleb2 and IEMOCAP dataset.

`/pretrained_networks`: The pre-trained usev network on the voxceleb2 dataset.

`/src`: The pre-training and training scripts.


## Training

We provided the pre-trained weights from the Voxceleb2 dataset, you can just generate the iemocap-mix dataset and train from the pre-trained checkpoint; or you can generate the pre-trained voxceleb2-mix dataset and start from the pre-training stage.