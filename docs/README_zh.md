# RASR：面向 PairUAV 的距离感知尺度恢复

这是 RASR（Range-Aware Scale Recovery）的公开实现。RASR 对应论文
`Range-Aware Scale Recovery for Monocular Metric Grounding`，用于 PairUAV
最后一米相对位姿估计任务。

PairUAV 的输入是一对有顺序的图像，输出是让无人机从当前视角移动到目标视角的
航向角和距离命令。论文把这个问题视为 monocular metric grounding：冻结的成对
几何模型已经提供了两张图像之间的相对结构，主要瓶颈是如何在相对误差指标下恢复
正确的米制尺度。

## 方法边界

RASR 明确区分两部分：

- **尺度恢复核心（scale-recovery core）**：冻结成对几何、422 维无元数据描述子、
  全局与距离感知校准、四个冻结距离候选，以及逐行凸组合。
- **PairUAV 榜单专用输出头（benchmark-specific output head）**：距离分桶、输出
  snapping、以及针对 PairUAV 标签和评分规则调出的常数。该部分用于复现榜单提交，
  不声明可直接迁移。

正式推理时，每一行预测只依赖这一对图像和固定参数。发布路径不使用测试集图优化、
batch 排序、隐藏邻居修正、测试集检索、全局分配、聚类或跨样本 bookkeeping。

## 在线结果

精确复现路径对应的 `submit.zip` SHA-256 为：

```text
2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77
```

对应线上结果：

```text
final_score          0.003189
distance_rel_error   0.003029
angle_rel_error      0.003350
```

## 两条复现路径

### Path A：从官方数据公开复现

这一路径从官方 PairUAV 数据开始，构建 pair table、提取 pair feature、训练公开距离
头和一个简单公开 heading baseline，最后生成合法的 `submit.zip`。

```bash
bash scripts/reproduce_from_scratch.sh \
  --data-root /path/to/pairuav-data \
  --output-dir outputs/from_scratch \
  --feature-mode smoke \
  --limit 4096 \
  --max-steps 50
```

这一路径用于透明地跑通端到端流程和独立实验，不承诺得到与线上归档提交完全一致的
hash 或分数。

### Path B：下载冻结中间结果精确复现

这是唯一需要外部 artifact bundle 的路径。bundle 中包含源码包、用于精确复现的
冻结逐行预测 CSV、校验和文件，以及归档 submit 包。

- 百度网盘：https://pan.baidu.com/s/1K1gDMw8mLJwAFC6jO-c9Fg
- 提取码：`t2c6`
- 文件名：`pairuav_lastmeter_complete_release_bundle.zip`
- bundle SHA-256：

```text
4680537d47c93f6b953d20fde6e55b260e4603f02743b06f6ad6900ec0ef729f
```

下载后，从完整 bundle 中解出 `pairuav_lastmeter_frozen_artifacts.zip`，再解压得到：

```text
/path/to/frozen_artifacts/
├── distance_head_a.csv
├── distance_head_b.csv
├── distance_head_c.csv
├── distance_head_d.csv
└── heading_predictions.csv
```

先校验 artifact：

```bash
python scripts/verify_frozen_artifacts.py \
  --artifact-root /path/to/frozen_artifacts \
  --manifest frozen_artifacts_manifest.json
```

再复现归档提交：

```bash
bash scripts/reproduce_from_models.sh \
  --data-root /path/to/pairuav-data \
  --feature-root /path/to/frozen_artifacts \
  --output-dir outputs/exact
```

成功后应得到：

```text
outputs/exact/submit.zip
```

其 SHA-256 应为：

```text
2f742f2eff83e535b96a8dbd46db370fa3ac0538a9f3e53b684d65c253b34b77
```

更多细节见英文文档：

- `README.md`
- `docs/REPRODUCTION.md`
- `docs/METHOD.md`
