import os, json, torch
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from dataclasses import dataclass
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_process import encode_sample


class LlavaJsonlIterable(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        jsonl_name: str,
        image_dir: str,
        shuffle: bool = False,     # 如需随机打乱，可设 True（需要一次读取索引，会多占一点点内存）
        seed: int = 0,
        max_samples: Optional[int] = None,  # 只取前 N 条用于小机调试
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.jsonl_path = self.data_dir / jsonl_name
        self.image_root = self.data_dir / image_dir
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples

    def _line_generator(self) -> Iterator[str]:
        """
        单进程逐行读取；在 DataLoader 多 worker 时由 __iter__ 做分片控制。
        """
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line

    def __iter__(self):
        """
        多 worker 分片：第 i 行只给 (i % num_workers == worker_id) 的 worker。
        """
        worker = get_worker_info()
        if worker is None:
            # 单 worker
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers
        # print(f"[INFO] worker_id={worker_id}, num_workers={num_workers}")
        cnt = 0
        for i, line in enumerate(self._line_generator()):
            if i % num_workers != worker_id:
                continue
            row = json.loads(line)
            conversations = row["conversations"]
            image_file = row["image"]
            image_path = os.path.join(self.image_root, image_file)

            # —— 注意：用 with 确保文件句柄及时释放，避免积累占内存/句柄 ——
            try:
                sample = encode_sample(conversations, image_path)
            except Exception as e:
                # 出错时跳过该样本，避免中断整个迭代
                print(f"[WARN] skip sample at line {i} due to: {e}")
                continue
            # print(f"[INFO] yielding sample #{cnt} (line={i}, image={image_file})")
            yield sample

            cnt += 1
            if self.max_samples is not None and cnt >= self.max_samples:
                break



"""
import os, json, torch
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, List
from dataclasses import dataclass
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from PIL import Image
import sys
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_process import encode_sample


class LlavaJsonlIterable(IterableDataset):
   
    def __init__(
        self,
        data_dir: str,
        jsonl_name: str,
        image_dir: str,
        shuffle: bool = False,     # True 时会构建行偏移索引（仅一遍扫描），内存开销很小
        seed: int = 0,
        max_samples: Optional[int] = None,  # 只取前 N 条用于小机调试
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.jsonl_path = self.data_dir / jsonl_name
        self.image_root = self.data_dir / image_dir
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples

        # 延迟初始化
        self._num_lines: Optional[int] = None       # 剔除空行后的有效行数
        self._offsets: Optional[List[int]] = None   # 每条有效样本所在文件的 byte 偏移（仅 shuffle=True 需要）

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL not found: {self.jsonl_path}")

    # ---------- 统计/索引构建 ----------

    def _scan_and_build(self, need_offsets: bool):
      
        self._num_lines = 0
        self._offsets = [] if need_offsets else None

        with open(self.jsonl_path, "rb") as f:  # 二进制读，保证 offset 精确
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                try:
                    s = line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    # 解码异常行直接跳过
                    continue
                if not s:
                    continue
                if need_offsets:
                    self._offsets.append(pos)
                self._num_lines += 1

    def _ensure_count(self):
        if self._num_lines is None:
            # 不需要偏移时只统计数量
            self._scan_and_build(need_offsets=False)

    def _ensure_offsets(self):
        if self._offsets is None or self._num_lines is None:
            # 需要偏移则同时统计数量
            self._scan_and_build(need_offsets=True)

    # ---------- 可选：PyTorch 可读长度 ----------

    def __len__(self) -> int:
       
        self._ensure_count()
        n = int(self._num_lines or 0)
        if self.max_samples is not None:
            n = min(n, int(self.max_samples))
        return n

    # ---------- 逐行生成器（低内存路径） ----------

    def _line_generator(self) -> Iterator[str]:
     
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    yield s

    # ---------- 随机访问：按偏移读取单行 ----------

    def _read_line_at(self, offset: int) -> Optional[str]:
        with open(self.jsonl_path, "rb") as f:
            f.seek(offset)
            raw = f.readline()
            try:
                s = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                return None
            return s if s else None

    # ---------- 迭代逻辑 ----------

    def __iter__(self):
  
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        rng = random.Random(self.seed)

        # --------- 路径 A：打乱 ---------
        if self.shuffle:
            self._ensure_offsets()
            total = self._num_lines or 0
            if total <= 0:
                return iter(())

            # 先得到全局索引 [0..total-1]，再按 worker 做“步长分片”，最后打乱
            # （分片后再打乱，保证各 worker 间样本不重叠）
            idx_all = range(total)
            idx_this_worker = list(idx_all)[worker_id::num_workers]
            rng.shuffle(idx_this_worker)

            # max_samples 限制（对每个 worker 局部生效，语义更直观）
            if self.max_samples is not None:
                idx_this_worker = idx_this_worker[: int(self.max_samples)]

            for k in idx_this_worker:
                offset = self._offsets[k]
                s = self._read_line_at(offset)
                if not s:
                    continue
                try:
                    row = json.loads(s)
                    conversations = row["conversations"]
                    image_file = row["image"]
                    image_path = os.path.join(self.image_root, image_file)
                    sample = encode_sample(conversations, image_path)
                except Exception as e:
                    print(f"[WARN] skip sample idx={k} due to: {e}")
                    continue
                yield sample
            return

        # --------- 路径 B：顺序流式（低内存） ---------
        cnt = 0
        for i, s in enumerate(self._line_generator()):
            if i % num_workers != worker_id:
                continue
            try:
                row = json.loads(s)
                conversations = row["conversations"]
                image_file = row["image"]
                image_path = os.path.join(self.image_root, image_file)
                sample = encode_sample(conversations, image_path)
            except Exception as e:
                print(f"[WARN] skip sample at line {i} due to: {e}")
                continue

            yield sample
            cnt += 1
            if self.max_samples is not None and cnt >= self.max_samples:
                break


"""