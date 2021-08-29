## News
- PSENet and PAN are included in [MMOCR](https://github.com/open-mmlab/mmocr).


## Recommended environment
```
Python 3.6+
Pytorch 1.7.0
torchvision 0.8.0
mmcv 1.3.11
editdistance
Polygon3
pyclipper
opencv-python 3.4.2.17
Cython
```

## Install
```shell script
pip install -r requirement.txt
./compile.sh
```
## Dataset
See [dataset](https://github.com/whai362/pan_pp.pytorch/tree/master/dataset).

## Training
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py ${CONFIG_FILE}
```
For example:
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py config/pan/pan_r18_ic15.py
```

## Test
```
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```
For example:
```shell script
python test.py config/pan/pan_r18_ic15.py checkpoints/pan_r18_ic15/checkpoint.pth.tar
```

## Speed
```shell script
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --report_speed
```
For example:
```shell script
python test.py config/pan/pan_r18_ic15.py checkpoints/pan_r18_ic15/checkpoint.pth.tar --report_speed
```

## Evaluation
See [eval](https://github.com/whai362/pan_pp.pytorch/tree/master/eval).

## Benchmark and model zoo
- [PAN](https://github.com/whai362/pan_pp.pytorch/tree/master/config/pan)
- [PSENet](https://github.com/whai362/pan_pp.pytorch/tree/master/config/psenet)
- [PAN++](https://github.com/whai362/pan_pp.pytorch/tree/master/config/pan_pp)
