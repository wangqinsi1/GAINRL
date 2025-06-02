import torch
import numpy as np
from torch.utils.data import Sampler


# ------------------------------------------------------------
# 通用工具：从排好序的索引列表里按高斯分布采样
# ------------------------------------------------------------
def gaussian_sample_list(sorted_indices, num_samples, center_index, std_dev):
    """返回抽样后的索引列表（已排序）。"""
    n = len(sorted_indices)
    idx_range = np.arange(n)
    probs = np.exp(-0.5 * ((idx_range - center_index) / std_dev) ** 2)
    probs /= probs.sum()
    chosen = np.random.choice(idx_range,
                              size=min(num_samples, n),
                              replace=False,
                              p=probs)
    chosen.sort()
    return [int(sorted_indices[i]) for i in chosen]

# ------------------------------------------------------------
# 核心 Sampler
# ------------------------------------------------------------
class GAINRLSampler(Sampler):
    """
    Args
    ----
    sort_list     : 你已经排好序(难度/奖励等)的样本索引序列
    subset_size   : 每个 epoch 实际抽多少条样本参与训练
    std           : 高斯分布的标准差(索引空间)，越大越“随机”
    device        : 用来计算 tanh 调整量所放的设备
    """
    def __init__(self,
                 sort_list,
                 subset_size,
                 n = 500,
                 alpha =2,
                 beta =0.5,
                 adj_max =1000,
                 adj_min =0,
                 std: float = 1000.0,
                 device: str | None = None):
        super().__init__(None)
        self.sort_list   = np.asarray(sort_list)
        self.subset_size = subset_size
        self.n           = n
        self.std         = std
        self.alpha       = alpha
        self.beta        = beta
        self.adj_min     = adj_min
        self.adj_max     = adj_max
        self.device      = torch.device(device
                              or ("cuda" if torch.cuda.is_available() else "cpu"))

        # ---- 下面是会随训练演化的状态 ----
        self.mean  = 0.0      # 采样中心(索引空间)
        self._subset = self._sample_subset()  # 当前 epoch 要用的索引列表

    # ---------- Sampler 接口 ----------
    def __iter__(self):
        """
        DataLoader 在每个 epoch 开头都会重新调用 __iter__，  
        所以直接把上一轮算好的 subset 返回即可。
        """
        # return iter(self._subset)
        yield self._subset

    def __len__(self):
        """让 DataLoader 能正常算 batch 数；也可直接返回 subset_size。"""
        return len(self._subset)

    # ---------- 公开 API ----------
    def update_after_epoch(self, records: list[dict]):
        """
        在 **一个 epoch 完毕**、拿到模型反馈(accuracy/angle)后调用。
        `records` 形如 [{'accuracy': 0.82, 'angle': 0.03}, ...]
        """
        if not records:  # 若本轮没收集到数据就直接跳过
            return

        # 1) 统计平均指标
        acc_mean = float(np.mean([r['accuracy'] for r in records]))
        ang_mean = float(np.mean([r['angle']    for r in records]))

        # 2) 用原公式计算采样中心的平移量
        acc = torch.tensor(acc_mean, dtype=torch.float32, device=self.device)
        ang = torch.tensor(ang_mean, dtype=torch.float32, device=self.device)

        adjustment = self.n * torch.tanh(self.alpha * (acc / 2 - self.beta)) \
                   + self.n * torch.tanh(self.alpha * ang)
        adjustment = torch.clamp(adjustment, self.adj_min, self.adj_max)

        # 3) 更新中心 & 重新抽样
        self.mean += adjustment.item()
        self._subset = self._sample_subset()

    # ---------- 内部方法 ----------
    def _sample_subset(self):
        """真正执行高斯采样，返回新的索引列表。"""
        return gaussian_sample_list(
            self.sort_list,
            self.subset_size,
            center_index=self.mean,
            std_dev=self.std,
        )