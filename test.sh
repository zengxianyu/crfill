python test.py \
	--batchSize 1 \
	--nThreads 1 \
	--name objrmv \
	--dataset_mode testimage \
	--image_dir ./datasets/places2sample1k_val/places2samples1k_crop256 \
	--mask_dir ./datasets/places2sample1k_val/places2samples1k_256_mask_square128 \
        --output_dir ./results \
	--model inpaint \
	--netG baseconv \
        --which_epoch latest \
	--load_baseg \
	$EXTRA
