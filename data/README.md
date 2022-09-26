## Project Structure


`/voxceleb2`: Scripts to generate 2-speaker mixture from Voxceleb2 for pretraining.

`/iemocap`: Scripts to generate 2/3-speaker mixture list for the USEV training.


## Dataset File Structure

Please prepare the dataset with the following file structure

	voxceleb2/
	  └── orig/
	    |── train/     	# The original train set contains .mp4 video
	    └── test/		# The original test set contains .mp4 video	

	wham_noise/
	  └── metadata/
	  └── test/
	  └── train/
	  └── val/

	iemocap/
