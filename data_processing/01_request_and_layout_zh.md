# 数据申请与目录结构

[English](01_request_and_layout.md)

本仓库不重新分发图像或标签。

## 所需来源

1. University-1652：从官方 University-1652 项目页面申请数据集，并遵守其使用条款。
2. PairUAV official data：从 benchmark release page 下载官方训练和测试数据。

## 期望目录结构

请在本仓库之外使用一个本地 data root：

```text
/path/to/pairuav-data/
├── University-1652/
└── PairUAV/
    ├── train/
    ├── test/
    └── metadata/
```

脚本接受 `--data-root` 参数，因此仓库本身不需要包含数据集文件。

## Smoke Mode

大多数脚本接受 `--limit N`，用于处理一个小前缀以验证代码。Smoke mode 用于检查 wiring、shapes、serialization 和 row-order stability。它不用于报告 benchmark performance。
