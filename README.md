

## env setup

Install miniconda (if you haven't already):
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


## running the demo

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