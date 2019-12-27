pip install mxnet-cu100
pip install gluoncv

MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python train_frcnn.py --gpus 0,1 --dataset change --network faster_rcnn_resnet50_v1b_custom
MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python train_frcnn.py --gpus 0,1 --dataset voc --network resnet50_v1b
MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python sample_train.py