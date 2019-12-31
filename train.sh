pip install mxnet-cu101
pip install gluoncv

scp -r -P 22778 yf@210.76.196.40:~/workplace/12-Det-ChangE ./

# GPU01
MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python train_frcnn.py --gpus 0,1 --batch-size 4 -j 16 --dataset change --network faster_rcnn_resnet50_v1b_custom

# GPU01
MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python train_frcnn.py --gpus 0,1 --batch-size 4 -j 16 --dataset change --network faster_rcnn_resnet101_v1d_custom --root /home/yf/workplace/12-Det-ChangE/data/5_CE2_50m_DOM_DEM_Sample_Data

