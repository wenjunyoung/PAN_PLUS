## News
- PSENet and PAN are included in [MMOCR](https://github.com/open-mmlab/mmocr).

## Introduction
Official Pytorch implementations of PSENet [1], PAN [2] and PAN++ [3].

[1] W. Wang, E. Xie, X. Li, W. Hou, T. Lu, G. Yu, and S. Shao. Shape robust text detection with progressive scale expansion network. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pages 9336–9345, 2019.<br>
[2] W. Wang, E. Xie, X. Song, Y. Zang, W. Wang, T. Lu, G. Yu, and C. Shen. Efficient and accurate arbitrary-shaped text detection with pixel aggregation network. In Proc. IEEE Int. Conf. Comp. Vis., pages 8440–8449, 2019.<br>
[3] W. Wang, E. Xie, X. Li, X. Liu, D. Liang, Z. Yang, T. Lu and C. Shen. PAN++: Towards Efficient and Accurate End-to-End Spotting of Arbitrarily-Shaped Text[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.


## Recommended environment
```
Python 3.6+
Pytorch 1.1.0
torchvision 0.3
mmcv 0.2.12
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


## Citation
```
@inproceedings{wang2019shape,
  title={Shape robust text detection with progressive scale expansion network},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Hou, Wenbo and Lu, Tong and Yu, Gang and Shao, Shuai},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9336--9345},
  year={2019}
}

@inproceedings{wang2019efficient,
  title={Efficient and accurate arbitrary-shaped text detection with pixel aggregation network},
  author={Wang, Wenhai and Xie, Enze and Song, Xiaoge and Zang, Yuhang and Wang, Wenjia and Lu, Tong and Yu, Gang and Shen, Chunhua},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={8440--8449},
  year={2019}
}
@article{wang2021pan++,
  title={PAN++: Towards Efficient and Accurate End-to-End Spotting of Arbitrarily-Shaped Text},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Liu, Xuebo and Liang, Ding and Zhibo, Yang and Lu, Tong and Shen, Chunhua},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
```

## License
This project is developed and maintained by [IMAGINE Lab@National Key Laboratory for Novel Software Technology, Nanjing University](https://cs.nju.edu.cn/lutong/ImagineLab.html).

<img src="logo.jpg" alt="IMAGINE Lab">

This project is released under the [Apache 2.0 license](https://github.com/whai362/pan_pp.pytorch/blob/master/LICENSE).
