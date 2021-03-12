cd src
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py tracking --exp_id crowdhuman --dataset crowdhuman --ltrb_amodal --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --num_epochs 140 --lr_step 90,120 --save_point 60,90 --gpus 0,1,2,3 --batch_size 64 --lr 2.5e-4 --num_workers 16 --clip_len 2 --max_frame_dist 10 --trades
cd ..