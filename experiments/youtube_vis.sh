cd src
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py tracking --exp_id ytvis --dataset youtube_vis --dataset_version train --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0,1,2,3 --load_model ../models/coco_seg.pth  --save_point 10,12,14,16 --lr_step 9 --num_epochs 16 --max_frame_dist 5  --batch_size 32 --clip_len 2 --trades --seg
# test
CUDA_VISIBLE_DEVICES=0 python test.py tracking --exp_id ytvis --dataset youtube_vis --dataset_version val --pre_hm  --pre_thresh 0.5 --inference --track_thresh 0.05 --load_model $model_path --gpus 0 --clip_len 2 --trades --box_nms 0.7 --save_results --seg
cd ..