import os
import math
from typing import Sequence
import numpy as np
from torch.utils.data import BatchSampler, Sampler, Dataset
from random import shuffle, choice
from copy import deepcopy
from sciforma.registry import DATASETS


import os
import math
import random
from typing import Sequence
import numpy as np
from torch.utils.data import BatchSampler, Sampler, Dataset
from sciforma.registry import DATASETS


# 修改 ar_batch_sampler.py
@DATASETS.register_module()
class ArXiVMixScaleBatchSampler(BatchSampler):
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 num_replicas: int = 1,
                 rank: int = 0,
                 drop_last: bool = True,
                 shuffle: bool = True,
                 seed: int = 42,
                 **kwargs) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # 预加载分组索引
        self._group_indices = self.dataset.meta_df.groupby(['bucket_h', 'bucket_w']).indices

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        # 1. 保证所有进程共享同一个随机数发生器
        rng = random.Random(self.seed + self.epoch)
        
        all_batches_for_this_rank = []
        
        # 2. 排序桶的 Key，保证所有 Rank 遍历顺序一致（防止 NCCL 死锁）
        sorted_keys = sorted(self._group_indices.keys())

        for key in sorted_keys:
            indices = self._group_indices[key].tolist()
            if len(indices) == 0: continue
            
            if self.shuffle:
                rng.shuffle(indices)

            # 3. 计算全局 Batch 大小
            global_bs = self.batch_size * self.num_replicas
            
            # 对当前桶执行对齐
            if self.drop_last:
                keep_len = (len(indices) // global_bs) * global_bs
                indices = indices[:keep_len]
            else:
                # 循环补齐，确保即便是不满的桶，每张卡也能分到同样多的 Batch
                needed = (math.ceil(len(indices) / global_bs) * global_bs) - len(indices)
                indices += indices[:needed]

            if len(indices) == 0: continue

            # 4. 全局分发：切分 Global Batch 并分发给 Rank
            for i in range(0, len(indices), global_bs):
                global_chunk = indices[i : i + global_bs]
                # 当前 Rank 拿走属于自己的部分 (例如 Rank 0 拿 0:BS)
                my_batch = global_chunk[self.rank * self.batch_size : (self.rank + 1) * self.batch_size]
                all_batches_for_this_rank.append(my_batch)

        # 5. 桶间随机打乱（所有 Rank 必须同步打乱，确保分辨率切换一致）
        if self.shuffle:
            rng.shuffle(all_batches_for_this_rank)
        
        # 调试信息 (仅主进程)
        if self.rank == 0 and self.epoch == 0:
            print(f"✅ Sampler initialized: Generated {len(all_batches_for_this_rank)} batches per rank.")

        return iter(all_batches_for_this_rank)

    def __len__(self) -> int:
        total = 0
        global_bs = self.batch_size * self.num_replicas
        for key in self._group_indices:
            n = len(self._group_indices[key])
            if self.drop_last:
                total += n // global_bs
            else:
                total += math.ceil(n / global_bs)
        return total
    
# @DATASETS.register_module()
# class ArXiVMixScaleBatchSampler(BatchSampler):
#     """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

#     Args:
#         sampler (Sampler): Base sampler.
#         dataset (Dataset): Dataset providing data information.
#         batch_size (int): Size of mini-batch.
#         drop_last (bool): If ``True``, the sampler will drop the last batch if
#             its size would be less than ``batch_size``.
#     """

#     def __init__(self,
#                  sampler: Sampler,
#                  dataset: Dataset,
#                  batch_size: int,
#                  drop_last: bool = True,
#                  num_replicas: int = 1,
#                 rank: int = 0,
#                  **kwargs) -> None:
#         if not isinstance(sampler, Sampler):
#             raise TypeError('sampler should be an instance of ``Sampler``, '
#                             f'but got {sampler}')
#         if not isinstance(batch_size, int) or batch_size <= 0:
#             raise ValueError('batch_size should be a positive integer value, '
#                              f'but got batch_size={batch_size}')
#         self.sampler = sampler
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         self.num_replicas = num_replicas
#         self.rank = rank
#         # buckets for each aspect ratio
#         self.resolution_buckets = [256, 512, 768, 1024]
#         self.aspect_ratio_helper = AspectRatioHelper()
#         self._buckets = self._build_buckets()

#     def _build_buckets(self):
#         _buckets = dict()
#         for resolution in self.aspect_ratio_helper.anchors_dict:
#             ar_dict = self.aspect_ratio_helper.anchors_dict[resolution]
#             for aspect_ratio in ar_dict:
#                 _buckets[f"{resolution}_{aspect_ratio}"] = []
#         return _buckets

#     def _get_batch_size(self, resolution):
#         config = {
#             "256": 16,
#             "512": 4,
#             "768": 2,
#             "1024": 1,
#         }
#         scale = config.get(resolution, 1)
#         return scale * self.batch_size

#     def __iter__(self) -> Sequence[int]:
#         print(f"Building aspect ratio buckets for rank {self.rank}...")
#         for idx in self.sampler:
#             data_info = self.dataset.get_data_info(idx)
            
#             w, h = int(data_info["bucket_w"]), int(data_info["bucket_h"]) # (w, h)
            
#             resolution_indice =[abs(math.sqrt(w * h) - res_val) for res_val in self.resolution_buckets]
#             real_resolution = self.resolution_buckets[resolution_indice.index(min(resolution_indice))]

#             aspect_ratio = data_info["aspect_ratio"]
            
            
#             resolution = str(int(real_resolution))
#             bucket = self._buckets[f"{resolution}_{aspect_ratio}"]
#             bucket.append(idx)
#             # yield a batch of indices in the same aspect ratio group
#             if len(bucket) == self._get_batch_size(resolution):
#                 yield bucket[:]
#                 del bucket[:]

#         # yield the rest data and reset the buckets
#         for key in self._buckets.keys():
#             bucket = self._buckets[key]
#             resolution = key.split('_')[0]
#             bs = self._get_batch_size(resolution)
#             while len(bucket) > 0:
#                 if len(bucket) <= bs:
#                     if not self.drop_last:
#                         yield bucket[:]
#                     bucket = []
#                 else:
#                     yield bucket[:bs]
#                     bucket = bucket[bs:]



ASPECT_RATIO_1024 = {
    '0.25': [512., 2048.], '0.26': [512., 1984.], '0.27': [512., 1920.], '0.28': [512., 1856.],
    '0.32': [576., 1792.], '0.33': [576., 1728.], '0.35': [576., 1664.], '0.4':  [640., 1600.],
    '0.42':  [640., 1536.], '0.48': [704., 1472.], '0.5': [704., 1408.], '0.52': [704., 1344.],
    '0.57': [768., 1344.], '0.6': [768., 1280.], '0.68': [832., 1216.], '0.72': [832., 1152.],
    '0.78': [896., 1152.], '0.82': [896., 1088.], '0.88': [960., 1088.], '0.94': [960., 1024.],
    '1.0':  [1024., 1024.], '1.07': [1024.,  960.], '1.13': [1088.,  960.], '1.21': [1088.,  896.],
    '1.29': [1152.,  896.], '1.38': [1152.,  832.], '1.46': [1216.,  832.], '1.67': [1280.,  768.],
    '1.75': [1344.,  768.], '2.0':  [1408.,  704.], '2.09':  [1472.,  704.], '2.4':  [1536.,  640.],
    '2.5':  [1600.,  640.], '2.89':  [1664.,  576.], '3.0':  [1728.,  576.], '3.11':  [1792.,  576.],
    '3.62':  [1856.,  512.], '3.75':  [1920.,  512.], '3.88':  [1984.,  512.], '4.0':  [2048.,  512.],
}

ASPECT_RATIO_768 = {
    '0.25': [384., 1536.], '0.26': [384., 1488.], '0.27': [384., 1440.], '0.28': [384., 1392.],
    '0.32': [432., 1344.], '0.33': [432., 1296.], '0.35': [432., 1248.], '0.4':  [480., 1200.],
    '0.42': [480., 1152.], '0.48': [528., 1104.], '0.5': [528., 1056.], '0.52': [528., 1008.],
    '0.57': [576., 1008.], '0.6':  [576., 960.], '0.68': [624., 912.], '0.72': [624., 864.],
    '0.78': [672., 864.], '0.82': [672., 816.], '0.88': [720., 816.], '0.94': [720., 768.],
    '1.0':  [768., 768.], '1.07': [768.,  720.], '1.13': [816., 720.], '1.21': [816.,  672.],
    '1.29': [864., 672.], '1.38': [864.,  624.], '1.46': [912., 624.], '1.67': [960.,  576.],
    '1.75': [1008.,  576.], '2.0':  [1056.,  528.], '2.09':  [1104.,  528.], '2.4':  [1152.,  480.],
    '2.5':  [1200.,  480.], '2.89':  [1248.,  432.], '3.0':  [1296.,  432.], '3.11':  [1344.,  432.],
    '3.62':  [1392.,  384.], '3.75':  [1440.,  384.], '3.88':  [1488., 384.], '4.0':  [1536.,  384.],
}

ASPECT_RATIO_512 = {
     '0.25': [256.0, 1024.0], '0.26': [256.0, 992.0], '0.27': [256.0, 960.0], '0.28': [256.0, 928.0],
     '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
     '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
     '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
     '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
     '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
     '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
     '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
     '2.5': [800.0, 320.0], '2.89': [832.0, 288.0], '3.0': [864.0, 288.0], '3.11': [896.0, 288.0],
     '3.62': [928.0, 256.0], '3.75': [960.0, 256.0], '3.88': [992.0, 256.0], '4.0': [1024.0, 256.0]
     }

ASPECT_RATIO_256 = {
     '0.25': [128.0, 512.0], '0.26': [128.0, 496.0], '0.27': [128.0, 480.0], '0.28': [128.0, 464.0],
     '0.32': [144.0, 448.0], '0.33': [144.0, 432.0], '0.35': [144.0, 416.0], '0.4': [160.0, 400.0],
     '0.42': [160.0, 384.0], '0.48': [176.0, 368.0], '0.5': [176.0, 352.0], '0.52': [176.0, 336.0],
     '0.57': [192.0, 336.0], '0.6': [192.0, 320.0], '0.68': [208.0, 304.0], '0.72': [208.0, 288.0],
     '0.78': [224.0, 288.0], '0.82': [224.0, 272.0], '0.88': [240.0, 272.0], '0.94': [240.0, 256.0],
     '1.0': [256.0, 256.0], '1.07': [256.0, 240.0], '1.13': [272.0, 240.0], '1.21': [272.0, 224.0],
     '1.29': [288.0, 224.0], '1.38': [288.0, 208.0], '1.46': [304.0, 208.0], '1.67': [320.0, 192.0],
     '1.75': [336.0, 192.0], '2.0': [352.0, 176.0], '2.09': [368.0, 176.0], '2.4': [384.0, 160.0],
     '2.5': [400.0, 160.0], '2.89': [416.0, 144.0], '3.0': [432.0, 144.0], '3.11': [448.0, 144.0],
     '3.62': [464.0, 128.0], '3.75': [480.0, 128.0], '3.88': [496.0, 128.0], '4.0': [512.0, 128.0]
}

ASPECT_RATIO_256_TEST = {
     '0.25': [128.0, 512.0], '0.28': [128.0, 464.0],
     '0.32': [144.0, 448.0], '0.33': [144.0, 432.0], '0.35': [144.0, 416.0], '0.4': [160.0, 400.0],
     '0.42': [160.0, 384.0], '0.48': [176.0, 368.0], '0.5': [176.0, 352.0], '0.52': [176.0, 336.0],
     '0.57': [192.0, 336.0], '0.6': [192.0, 320.0], '0.68': [208.0, 304.0], '0.72': [208.0, 288.0],
     '0.78': [224.0, 288.0], '0.82': [224.0, 272.0], '0.88': [240.0, 272.0], '0.94': [240.0, 256.0],
     '1.0': [256.0, 256.0], '1.07': [256.0, 240.0], '1.13': [272.0, 240.0], '1.21': [272.0, 224.0],
     '1.29': [288.0, 224.0], '1.38': [288.0, 208.0], '1.46': [304.0, 208.0], '1.67': [320.0, 192.0],
     '1.75': [336.0, 192.0], '2.0': [352.0, 176.0], '2.09': [368.0, 176.0], '2.4': [384.0, 160.0],
     '2.5': [400.0, 160.0], '3.0': [432.0, 144.0],
     '4.0': [512.0, 128.0]
}

ASPECT_RATIO_512_TEST = {
     '0.25': [256.0, 1024.0], '0.28': [256.0, 928.0],
     '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
     '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
     '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
     '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
     '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
     '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
     '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
     '2.5': [800.0, 320.0], '3.0': [864.0, 288.0],
     '4.0': [1024.0, 256.0]
     }

ASPECT_RATIO_768_TEST = {
    '0.25': [384., 1536.], '0.28': [384., 1392.],
    '0.32': [432., 1344.], '0.33': [432., 1296.], '0.35': [432., 1248.], '0.4':  [480., 1200.],
    '0.42': [480., 1152.], '0.48': [528., 1104.], '0.5': [528., 1056.], '0.52': [528., 1008.],
    '0.57': [576., 1008.], '0.6':  [576., 960.], '0.68': [624., 912.], '0.72': [624., 864.],
    '0.78': [672., 864.], '0.82': [672., 816.], '0.88': [720., 816.], '0.94': [720., 768.],
    '1.0':  [768., 768.], '1.07': [768.,  720.], '1.13': [816., 720.], '1.21': [816.,  672.],
    '1.29': [864., 672.], '1.38': [864.,  624.], '1.46': [912., 624.], '1.67': [960.,  576.],
    '1.75': [1008.,  576.], '2.0':  [1056.,  528.], '2.09':  [1104.,  528.], '2.4':  [1152.,  480.],
    '2.5':  [1200.,  480.], '3.0':  [1296.,  432.],
    '4.0':  [1536.,  384.],
}

ASPECT_RATIO_1024_TEST = {
    '0.25': [512., 2048.], '0.28': [512., 1856.],
    '0.32': [576., 1792.], '0.33': [576., 1728.], '0.35': [576., 1664.], '0.4':  [640., 1600.],
    '0.42':  [640., 1536.], '0.48': [704., 1472.], '0.5': [704., 1408.], '0.52': [704., 1344.],
    '0.57': [768., 1344.], '0.6': [768., 1280.], '0.68': [832., 1216.], '0.72': [832., 1152.],
    '0.78': [896., 1152.], '0.82': [896., 1088.], '0.88': [960., 1088.], '0.94': [960., 1024.],
    '1.0':  [1024., 1024.], '1.07': [1024.,  960.], '1.13': [1088.,  960.], '1.21': [1088.,  896.],
    '1.29': [1152.,  896.], '1.38': [1152.,  832.], '1.46': [1216.,  832.], '1.67': [1280.,  768.],
    '1.75': [1344.,  768.], '2.0':  [1408.,  704.], '2.09':  [1472.,  704.], '2.4':  [1536.,  640.],
    '2.5':  [1600.,  640.], '3.0':  [1728.,  576.],
    '4.0':  [2048.,  512.],
}

class AspectRatioHelper:
    def __init__(self, test=False) -> None:
        self.anchor2resolution, self.anchor2ar, self.anchors, self.anchors_dict = self._build_data(test)
        self.test = test
        self.anchors = np.array(self.anchors)
        self.anchors_area = self.anchors[:, 0] * self.anchors[:, 1]
        self.max_size = 1024

    def _build_data(self, test):
        anchor2resolution = []
        anchor2ar = []
        anchors = []
        anchors_dict = dict()
        if test:
            ar_list = ['256_TEST', "512_TEST", "768_TEST", "1024_TEST"]
        else:
            ar_list = ['256', "512", "768", "1024"]
        for ar_name in ar_list:
            sample = build_aspect_ratio_dict(ar_name)
            resolution = ar_name.split('_')[0]
            anchors_dict[resolution] = sample
            for ar in sample.keys():
                hw = sample[ar]
                anchor2resolution.append(resolution)
                anchor2ar.append(ar)
                anchors.append(hw)
        return anchor2resolution, anchor2ar, anchors, anchors_dict

    def rescale_size(self, h, w):
        if h * w <= self.max_size * self.max_size:
            return h, w
        rescale = math.sqrt(h * w / (self.max_size * self.max_size))
        return h / rescale, w / rescale

    def compute_dis(self, h, w):
        h, w = self.rescale_size(h, w)
        min_h = np.minimum(self.anchors[:, 0], h)
        min_w = np.minimum(self.anchors[:, 1], w)
        iou_area = min_h * min_w
        sample_area = h * w
        iou = iou_area / (self.anchors_area + sample_area - iou_area)
        return iou

    def search(self, h, w):
        iou = self.compute_dis(h, w)
        index = np.argmax(iou)
        resolution = self.anchor2resolution[index]
        ar = self.anchor2ar[index]
        result = {
            "resolution": resolution,
            "aspect_ratio": ar,
            "anchor": self.anchors[index].tolist(),
            "IoU": iou[index],
        }
        return result
    
    def get_h_w(self, resolution, aspect_ratio):
        as_dict = self.anchors_dict.get(resolution, None)
        if as_dict is None:
            return None
        hw = as_dict.get(aspect_ratio, None)
        height, width = hw
        return height, width

def build_aspect_ratio_dict(name):
    if name == "256":
        return ASPECT_RATIO_256
    elif name == "512":
        return ASPECT_RATIO_512
    elif name == "768":
        return ASPECT_RATIO_768
    elif name == "1024":
        return ASPECT_RATIO_1024
    elif name == "256_TEST":
        return ASPECT_RATIO_256_TEST
    elif name == "512_TEST":
        return ASPECT_RATIO_512_TEST
    elif name == "768_TEST":
        return ASPECT_RATIO_768_TEST
    elif name == "1024_TEST":
        return ASPECT_RATIO_1024_TEST
    else:
        raise ValueError(name)

def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)

if __name__ == "__main__":
    aspect_ratio_helper = AspectRatioHelper()
    h = 612 #694 #1920
    w = 694 #612 #1080
    result = aspect_ratio_helper.search(h, w)
    print(result)
