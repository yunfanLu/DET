# -*- coding: utf-8 -*-
# @Time    : 2019/12/30 13:54
# @Author  : yunfan

import os
import numpy as np
import mxnet as mx

def main():
    root = '/home/yf/workplace/12-Det-ChangE/data/5_CE2_50m_DOM_DEM_Sample_Data/CE2_GRAS_50m_F003_21N135W_A'
    dom_tif_path = 'L0_2048_7168_3072_8192_CE2_GRAS_DOM_50m_F003_21N135W_A.tif'
    nd_image = mx.image.imread(os.path.join(root, dom_tif_path))
    print(nd_image.shape)


if __name__ == '__main__':
    import pudb
    pudb.set_trace()
    main()