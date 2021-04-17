# Dataset preparation

## MOT 2017 Dataset

MOT is the main dataset for training and evaluating the TraDeS tracker. 

We have packed the dataset preprocessing code as a script:

~~~
    cd $TraDeS_ROOT/src/tools/
    sh get_mot_17.sh
~~~

The script includes:

- Download and unzip the dataset from [MOT17 website](https://motchallenge.net/data/MOT17/).
- Convert it into COCO format using `tools/convert_mot_to_coco.py`.
- Create the half-half train/ val set described in the paper.
- The output data structure should be:

  ~~~
  ${TraDeS_ROOT}
  |-- data
  `-- |-- mot17
      `-- |--- train
          |   |--- MOT17-02-FRCNN
          |   |    |--- img1
          |   |    |--- gt
          |   |    |   |--- gt.txt
          |   |    |   |--- gt_train_half.txt
          |   |    |   |--- gt_val_half.txt
          |   |    |--- det
          |   |    |   |--- det.txt
          |   |    |   |--- det_train_half.txt
          |   |    |   |--- det_val_half.txt
          |   |--- ...
          |--- test
          |   |--- MOT17-01-FRCNN
          |---|--- ...
          `---| annotations
              |--- train_half.json
              |--- val_half.json
              |--- train.json
              `--- test.json
  ~~~

## nuScenes Dataset

nuScenes is used for training and evaluating 3D object tracking.

- Download the dataset from [nuScenes website](https://www.nuscenes.org/download). You only need to download the "Keyframe blobs", and only need the images data. You also need to download the maps and all metadata to make the nuScenes API happy.
According to the current website version, the data is under Full dataset (v1.0) -- Trainval. You should download "Metadata" and "Keyframe blobs only for part * [US]". The estimated data size is around 60G or less.

- Unzip, rename, and place (or symlink) the data as below. You will need to merge folders from different zip files.

  ~~~
  ${TraDeS_ROOT}
  |-- data
  `-- |-- nuscenes
      `-- |-- v1.0-trainval
          |   |-- samples
          |   |   |-- CAM_BACK
          |   |   |   | -- xxx.jpg
          |   |   |-- CAM_BACK_LEFT
          |   |   |-- CAM_BACK_RIGHT
          |   |   |-- CAM_FRONT
          |   |   |-- CAM_FRONT_LEFT
          |   |   |-- CAM_FRONT_RIGHT
          |-- |-- |-- maps
          |-- |-- |-- v1.0-trainval_meta
          `-- annotations
          |-- |-- train.json
          |-- |-- val.json
          |-- |-- test.json
  
  ~~~

- Run `python tools/convert_nuScenes.py` to convert the annotation into COCO format. It will create `train.json`, `val.json`, `test.json` under `data/nuscenes/annotations`. nuScenes API is required for running the data preprocessing.

## CrowdHuman Dataset

CrowdHuman is used for pretraining the MOT model. Only the training set is used.

- Download the dataset from [its website](https://www.crowdhuman.org/download.html).

- Unzip and place (or symlink) the data as below. You will need to merge folders from different zip files.

  ~~~
  ${TraDeS_ROOT}
  |-- data
  `-- |-- crowdhuman
      |-- |-- CrowdHuman_train
      |   |   |-- Images
      |-- |-- CrowdHuman_val
      |   |   |-- Images
      |-- |-- annotation_train.odgt
      |-- |-- annotation_val.odgt
  ~~~

- Run `python tools/convert_crowdhuman_to_coco.py` to convert the annotation into COCO format. It will create `train.json`, `val.json` under `data/crowdhuman/annotations`.

## Youtube-VIS Dataset

- Download the dataset from [website](https://competitions.codalab.org/competitions/20128#participate-get_data).

- Converted annotations: [train.json](https://drive.google.com/file/d/1cKIwLkUfmVMUWVUiRY3inQ8W6AO4ETLb/view?usp=sharing) and [val.json](https://drive.google.com/file/d/1iquYUDok2Eksnb-CwadqY22V7G8Iz8N3/view?usp=sharing).

-- coco pretrained model: [coco_seg.pth](https://drive.google.com/file/d/1zhTkO2KFVB72jgqiuaZsKN8S0ax7Zf19/view?usp=sharing)

  ~~~
  ${TraDeS_ROOT}
  |-- data
  `-- |-- youtube_vis
      |-- |-- train/
      |-- |-- val/
      |-- |-- annotations/
      |-- |-- |-- train.json
      |-- |-- |-- val.json
  ~~~



## References
Please cite the corresponding references if you use the datasets.

~~~
  @article{MOT16,
    title = {{MOT}16: {A} Benchmark for Multi-Object Tracking},
    shorttitle = {MOT16},
    url = {http://arxiv.org/abs/1603.00831},
    journal = {arXiv:1603.00831 [cs]},
    author = {Milan, A. and Leal-Taix\'{e}, L. and Reid, I. and Roth, S. and Schindler, K.},
    month = mar,
    year = {2016},
    note = {arXiv: 1603.00831},
    keywords = {Computer Science - Computer Vision and Pattern Recognition}
  }

  @article{shao2018crowdhuman,
    title={Crowdhuman: A benchmark for detecting human in a crowd},
    author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
    journal={arXiv:1805.00123},
    year={2018}
  }

  @inproceedings{nuscenes2019,
  title={{nuScenes}: A multimodal dataset for autonomous driving},
  author={Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and Giancarlo Baldan and Oscar Beijbom},
  booktitle={CVPR},
  year={2020}
  }
  
  @inproceedings{yang2019video,
  title={Video instance segmentation},
  author={Yang, Linjie and Fan, Yuchen and Xu, Ning},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5188--5197},
  year={2019}
}
~~~