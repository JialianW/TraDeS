cd src
# test
CUDA_VISIBLE_DEVICES=0 python test.py tracking,ddd --exp_id nuScenes_3Dtracking --dataset nuscenes --pre_hm --track_thresh 0.1 --gpus 0 --inference --load_model ../models/nuscenes.pth --clip_len 2 --trades
cd ..