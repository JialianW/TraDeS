cd src
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py tracking --exp_id mot17_fulltrain --dataset mot --dataset_version 17trainval --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0,1,2,3,4,5,6,7 --load_model ../models/crowdhuman.pth --clip_len 3 --max_frame_dist 10  --batch_size 32 --trades
# test
CUDA_VISIBLE_DEVICES=0 python test.py tracking --exp_id mot17_fulltrain --dataset mot --dataset_version 17test --pre_hm --ltrb_amodal --pre_thresh 0.5 --inference --clip_len 3 --track_thresh 0.4 --gpus 0 --trades --resume
cd ..