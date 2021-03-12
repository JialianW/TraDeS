cd src
# test
CUDA_VISIBLE_DEVICES=0 python test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --pre_thresh 0.5 --inference --track_thresh 0.4 --load_model ../models/mot_half.pth --gpus 0 --clip_len 3 --trades
cd ..