# Installation


The code was tested with [Anaconda](https://www.anaconda.com/download) Python 3.6, CUDA 10.0, and [PyTorch]((http://pytorch.org/)) v1.3.
Check your gcc version by 'gcc -v'. gcc version may need to be higher than v4.8 in order to compile the DCNv2 package. We tested the code with both gcc v5.4.0 and v8.4.0.
After installing Anaconda:

0. [Optional but highly recommended] create a new conda environment. 

    ~~~
    conda create --name trades python=3.6
    ~~~
    And activate the environment.
    
    ~~~
    conda activate trades
    ~~~

1. Install PyTorch:

    ~~~
    conda install pytorch=1.3.1 torchvision=0.4.2 cudatoolkit=10.0.130 -c pytorch
    ~~~
    

2. Install COCOAPI:

    ~~~
    pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    ~~~

3. Clone this repo:

    ~~~
    git clone https://github.com/JialianW/TraDeS.git
    ~~~

4. Install the requirements

    ~~~
    cd $TraDeS_ROOT
    pip install -r requirements.txt
    ~~~
    
    
5. Compile deformable convolutional (Successuflly compiled with both gcc v5.4.0 and v8.4.0. gcc version should be higher than v4.8).

    ~~~
    cd $TraDeS_ROOT/src/lib/model/networks/DCNv2
    . make.sh
    ~~~
   (modified from [DCNv2](https://github.com/CharlesShang/DCNv2/))

Note: We found the nuScenes and MOT dataset API versions are not compatible, you can switch between them by running 'sh mot_switch_version.sh' (for MOT experiments)
or 'sh nuscenes_switch_version.sh' (for nuScenes experiments). The default installed versions are for MOT dataset.