## Usage

Run each file one by one. It is to note that cropping face from the original iemocap is hard, you need to adjust the --num_failed_det; the face detector S3FD/MTCNN; the face detector confidence threshold for different videos. If you need assist in generating the data, please email us for assistance.

## Generated dataset file structure

	iemocap/
	  └── audio_clean/
	    |── train/
	    |── val/
	    └── test/
	  └── visual_embedding/lip/	
	    |── train/
	    |── val/
	    └── test/
	  └── audio_mixture/
	    └──2_mix_min_sparse_noise/ 
	      |── train/
	      |── val/
	      |── test/
	      └──mixture_data_list_mix.csv
	    └──3_mix_min_sparse_noise/ 
	      |── train/
	      |── val/
	      └── test/
