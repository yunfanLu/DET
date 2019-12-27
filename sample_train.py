# -*- coding: utf-8 -*-
# @Time    : 2019/12/27 10:07
# @Author  : yunfan

# -*- coding: utf-8 -*-
# @Time    : 2019/4/23 18:55
# @Author  : yunfan

import matplotlib.pyplot as plt

plt.switch_backend('agg')
import time
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
from ChangEDataset import ChangEDET

moon_path = '/home/yf/data/7-moon/5_CE2_50m_DOM_DEM_Sample_Data'
dataset = ChangEDET(moon_path)
classes = dataset.CLASSES

net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)

net.reset_class(classes)

net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes,
                              pretrained_base=False, transfer='voc')


def get_dataloader(net, train_dataset, data_shape, batch_size, num_workers):
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    return train_loader


train_data = get_dataloader(net, dataset, 512, 8, 0)

ctx = [mx.gpu(0),  mx.gpu(1)]

net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(
    net.collect_params(), 'sgd',
    {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9})

mbox_loss = gcv.loss.SSDMultiBoxLoss()
ce_metric = mx.metric.Loss('CrossEntropy')
smoothl1_metric = mx.metric.Loss('SmoothL1')

for epoch in range(0, 30):
    ce_metric.reset()
    smoothl1_metric.reset()
    tic = time.time()
    btic = time.time()
    net.hybridize(static_alloc=True, static_shape=True)
    for i, batch in enumerate(train_data):
        batch_size = batch[0].shape[0]
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
        with autograd.record():
            cls_preds = []
            box_preds = []
            for x in data:
                cls_pred, box_pred, _ = net(x)
                cls_preds.append(cls_pred)
                box_preds.append(box_pred)
            sum_loss, cls_loss, box_loss = mbox_loss(
                cls_preds, box_preds, cls_targets, box_targets)
            autograd.backward(sum_loss)
        trainer.step(1)
        ce_metric.update(0, [l * batch_size for l in cls_loss])
        smoothl1_metric.update(0, [l * batch_size for l in box_loss])
        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        print('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
            epoch, i, batch_size / (time.time() - btic), name1, loss1, name2, loss2))
        btic = time.time()

    net.save_parameters(f'ssd_512_mobilenet1.0_moon_{epoch}.params')
