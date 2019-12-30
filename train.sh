pip install mxnet-cu100
pip install gluoncv

MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python train_frcnn.py --gpus 0,1 --batch-size 4 -j 16 --dataset change --network faster_rcnn_resnet50_v1b_custom

