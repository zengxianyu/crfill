##BSIZE0=48 # stage coarse
BSIZE=96 # 96:64G
BSIZE0=$((BSIZE/2))
NWK=16
PREFIX="--dataset_mode_train trainimage \
--name objrmv_finetune \
--gpu_ids 0,1 \
--dataset_mode_val valimage \
--train_image_dir ./datasets/places/places2 \
--train_image_list ./datasets/places/train_example.txt \
--path_objectshape_list ./datasets/object_shapes.txt \
--path_objectshape_base ./datasets/object_masks \
--val_image_dir ./datasets/places2sample1k_val/places2samples1k_crop256 \
--val_image_list ./datasets/places2sample1k_val/files.txt \
--val_mask_dir ./datasets/places2sample1k_val/places2samples1k_256_mask_square128 \
--no_vgg_loss \
--no_ganFeat_loss \
--gpu_ids 0,1 \
--load_size 640 \
--crop_size 256 \
--model arrange \
--netG twostagend \
--baseG baseconv \
--norm_type 1 \
--netD deepfill \
--load_pretrained_g ./checkpoints/objrmv_finetune/pretrained_net_G.pth \
--load_pretrained_d ./checkpoints/objrmv_finetune/pretrained_net_D_aux.pth \
--load_base_d ./checkpoints/objrmv_finetune/pretrained_net_D.pth \
--lambda_ref 0.5 \
--lambda_l1 1 \
--preprocess_mode scale_shortside_and_crop"
python train.py \
	${PREFIX} \
	--batchSize ${BSIZE0} \
	--nThreads ${NWK} \
	--update_part all \
	--niter 10 \
	${EXTRA}
