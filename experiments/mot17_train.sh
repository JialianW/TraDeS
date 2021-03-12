cd src
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halftrain --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0,1,2,3 --load_model ../models/crowdhuman.pth  --save_point 40,50,60 --max_frame_dist 10  --batch_size 32 --clip_len 3 --trades
cd ..