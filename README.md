# AllTracker: Efficient Dense Point Tracking at High Resolution

**[[Paper](https://arxiv.org/abs/2506.07310)] [[Project Page](https://alltracker.github.io/)]**

## Env setup

Install miniconda:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init
```

Set up a fresh conda environment for AllTracker:

```
conda create -n alltracker python=3.12.8
conda activate alltracker
pip install -r requirements.txt
```


## Running the demo

Download the model:

```
download_reference_model.sh
tar -xvf alltracker_reference.tar.gz
```

Download the sample video:
```
cd demo_video
sh download_video.sh
cd ..
```

Run the demo
```
python demo.py
```


## Training code

Coming soon! (After I get back from CVPR.)


## Citation

If you use this code for your research, please cite:

**Adam W. Harley, Yang You, Xinglong Sun, Yang Zheng, Nikhil Raghuraman, Yunqi Gu, Sheldon Liang, Wen-Hsuan Chu, Achal Dave, Pavel Tokmakov, Suya You, Rares Ambrus, Katerina Fragkiadaki, Leonidas J. Guibas. AllTracker: Efficient Dense Point Tracking at High Resolution. arXiv 2025.**

Bibtex:
```
@inproceedings{harley2025alltracker,
author    = {Adam W. Harley and Yang You and Xinglong Sun and Yang Zheng and Nikhil Raghuraman and Yunqi Gu and Sheldon Liang and Wen-Hsuan Chu and Achal Dave and Pavel Tokmakov and Suya You and Rares Ambrus and Katerina Fragkiadaki and Leonidas J. Guibas},
title     = {All{T}racker: {E}fficient Dense Point Tracking at High Resolution}
booktitle = {arXiv},
year      = {2025}
}
```
