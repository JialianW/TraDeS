# Track to Detect and Segment: An Online Multi-Object Tracker (CVPR 2021)

[comment]: <> (> [**Track to Detect and Segment: An Online Multi-Object Tracker**]&#40;http://arxiv.org/abs/2004.01177&#41;,            )
[**Track to Detect and Segment: An Online Multi-Object Tracker**](https://arxiv.org/abs/2103.08808)  
Jialian Wu, Jiale Cao, Liangchen Song, Yu Wang, Ming Yang, Junsong Yuan        
In CVPR, 2021. [[Paper]](https://arxiv.org/pdf/2103.08808.pdf) [[Project Page]](https://jialianwu.com/projects/TraDeS.html) [Demo [(YouTube)](https://www.youtube.com/watch?v=oGNtSFHRZJAl) [(bilibili)](https://www.bilibili.com/video/BV12U4y1p7wg)]

Many thanks to [CenterTrack](https://github.com/xingyizhou/CenterTrack) authors for their great framework!

<p align="left"> <img src='https://github.com/JialianW/homepage/blob/master/images/TraDeS_demo.gif?raw=true' align="center" width="400px">

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Run Demo
Before run the demo, first download our trained models:
[CrowdHuman model](https://drive.google.com/file/d/1pljgwSecg50OhCTc2yCEhEBY3AwvPFlp/view?usp=sharing) (2D tracking),
[MOT model](https://drive.google.com/file/d/18DQi6LqFuO7_2QObvZSNK2y_F8yXT17p/view?usp=sharing) (2D tracking) or [nuScenes model](https://drive.google.com/file/d/1PHcDPIvb6owVuMZKR_YieyYN12IhbQLl/view?usp=sharing) (3D tracking). 
Then, put the models in `TraDeS_ROOT/models/` and `cd TraDeS_ROOT/src/`. **The demo result will be saved as a video in `TraDeS_ROOT/results/`.**

### *2D Tracking Demo*
**Demo for a video clip from MOT dataset**: Run the demo (using the [MOT model](https://drive.google.com/file/d/18DQi6LqFuO7_2QObvZSNK2y_F8yXT17p/view?usp=sharing)):

    python demo.py tracking --dataset mot --load_model ../models/mot_half.pth --demo ../videos/mot_mini.mp4 --pre_hm --ltrb_amodal --pre_thresh 0.5 --track_thresh 0.4 --inference --clip_len 3 --trades --save_video --resize_video --input_h 544 --input_w 960

**Demo for a video clip which we randomly selected from YouTube**: Run the demo (using the [CrowdHuman model](https://drive.google.com/file/d/1pljgwSecg50OhCTc2yCEhEBY3AwvPFlp/view?usp=sharing)):

    python demo.py tracking --load_model ../models/crowdhuman.pth --num_class 1 --demo ../videos/street_2d.mp4 --pre_hm --ltrb_amodal --pre_thresh 0.5 --track_thresh 0.5 --inference --clip_len 2 --trades --save_video --resize_video --input_h 480 --input_w 864

**Demo for your own video or image folder**: Please specify the file path after `--demo` and run (using the [CrowdHuman model](https://drive.google.com/file/d/1pljgwSecg50OhCTc2yCEhEBY3AwvPFlp/view?usp=sharing)):

    python demo.py tracking --load_model ../models/crowdhuman.pth --num_class 1 --demo $path to your video or image folder$ --pre_hm --ltrb_amodal --pre_thresh 0.5 --track_thresh 0.5 --inference --clip_len 2 --trades --save_video --resize_video --input_h $your_input_h$ --input_w $your_input_w$


(Some Notes: (i) For 2D tracking, the models are only used for person tracking, since our method is only trained on CrowdHuman or MOT. You may train a model on COCO or your own dataset for multi-category 2D object tracking. 
(ii) `--clip_len` is set to 3 for MOT; otherwise, it should be 2. You may refer to our paper for this detail. (iii) The CrowdHuman model is more able to generalize to real world scenes than the MOT model. Note that both datasets are in non-commercial licenses.
(iii) `input_h` and `input_w` shall be evenly divided by 32.)

### *3D Tracking Demo*
**Demo for a video clip from nuScenes dataset**: Run the demo (using the [nuScenes model](https://drive.google.com/file/d/1PHcDPIvb6owVuMZKR_YieyYN12IhbQLl/view?usp=sharing)):

    python demo.py tracking,ddd --dataset nuscenes --load_model ../models/nuscenes.pth --demo ../videos/nuscenes_mini.mp4 --pre_hm --track_thresh 0.1 --inference --clip_len 2 --trades --save_video --resize_video --input_h 448 --input_w 800 --test_focal_length 633

(You will need to specify test_focal_length for monocular 3D tracking demo to convert the image coordinate system back to 3D. The value 633 is half of a typical focal length (~1266) in nuScenes dataset in input resolution 1600x900. The mini demo video is in an input resolution of 800x448, so we need to use a half focal length. You don't need to set the test_focal_length when testing on the original nuScenes data.)

You can also refer to [CenterTrack](https://github.com/xingyizhou/CenterTrack) for the usage of webcam demo (code is available in this repo, but we have not tested yet).

## Benchmark Evaluation and Training

Please refer to [Data.md](readme/DATA.md) for dataset preparation.

### *2D Object Tracking*

| MOT17 Val                  | MOTA↑  |IDF1↑|IDS↓|
|-----------------------|----------|----------|----------|
| Our Baseline         |64.8|59.5|1055|
| [CenterTrack](https://arxiv.org/pdf/2004.01177.pdf)         |66.1|64.2|528|
| [TraDeS (ours)](experiments/mot17_test.sh)  |**68.2**|**71.7**|**285**|

**Test on MOT17 validation set:** Place the [MOT model](https://drive.google.com/file/d/18DQi6LqFuO7_2QObvZSNK2y_F8yXT17p/view?usp=sharing) in $TraDeS_ROOT/models/ and run:

    sh experiments/mot17_test.sh

**Train on MOT17 halftrain set:** Place the [pretrained model](https://drive.google.com/file/d/1pljgwSecg50OhCTc2yCEhEBY3AwvPFlp/view?usp=sharing) in $TraDeS_ROOT/models/ and run:

    sh experiments/mot17_train.sh

## *3D Object Tracking* 

| nuScenes Val                  | AMOTA↑|AMOTP↓|IDSA↓|
|-----------------------|----------|----------|----------|
| Our Baseline         |4.3|1.65|1792|
| [CenterTrack](https://arxiv.org/pdf/2004.01177.pdf)         |6.8|1.54|813|
| [TraDeS (ours)](experiments/nuScenes_test.sh) |**11.8**|**1.48**|**699**|

**Test on nuScenes validation set:** Place the [nuScenes model](https://drive.google.com/file/d/1PHcDPIvb6owVuMZKR_YieyYN12IhbQLl/view?usp=sharing) in $TraDeS_ROOT/models/. You need to change the MOT and nuScenes dataset API versions due to their conflicts. The default installed versions are for MOT dataset.  For experiments on nuScenes dataset, please run:

    sh nuscenes_switch_version.sh

    sh experiments/nuScenes_test.sh

To switch back to the API versions for MOT experiments, you can run:

    sh mot_switch_version.sh

**Train on nuScenes train set:** Place the [pretrained model](https://drive.google.com/file/d/1jGDrQ5I3ZxyGoep79egcT9MI3JM1ZKhG/view?usp=sharing) in $TraDeS_ROOT/models/ and run:
    
    sh experiments/nuScenes_train.sh

## *Train on Static Images*
We follow [CenterTrack](https://arxiv.org/pdf/2004.01177.pdf) which uses CrowdHuman to pretrain 2D object tracking model. Only the training set is used.

    sh experiments/crowdhuman.sh

The trained model is available at [CrowdHuman model](https://drive.google.com/file/d/1pljgwSecg50OhCTc2yCEhEBY3AwvPFlp/view?usp=sharing).


## Citation
If you find it useful in your research, please consider citing our paper as follows:

    @inproceedings{Wu2021TraDeS,
    title={Track to Detect and Segment: An Online Multi-Object Tracker},
    author={Wu, Jialian and Cao, Jiale and Song, Liangchen and Wang, Yu and Yang, Ming and Yuan, Junsong},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021}}