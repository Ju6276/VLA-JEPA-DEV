# 单独训练与评估空间目标预测器

该实验只训练 `SpatialGoalPredictor`，检查它能否利用当前观测、指令、state 和真实历史，预测独立示范片段中的未来空间特征。输入来自冻结的 V-JEPA 2.1 与 Qwen，训练目标是未来图像的固定空间网格，不需要目标框、mask 或人工 subgoal。

实验分为冻结特征提取和空间 head 训练两步，方便复用相同输入、比较设置并降低重复编码开销。**这是模块诊断流程；正常 policy 训练仍在一次训练中联合优化各模块，不增加必需的预训练阶段。** 完整方法见 [空间目标与动作选择](spatial_goals.md)。

## 1. 提取冻结特征

在主 README 的环境中，设置对应动作接口的数据和预训练模型路径：

```bash
export DATA_ROOT="/path/to/lerobot/datasets"
export QWEN_MODEL="/path/to/Qwen3-VL-2B-Instruct"
export VJEPA21_CKPT="/path/to/vjepa2_1_vitl_dist_vitG_384.pt"
export PROBE_ROOT="/path/to/experiments/simple_spatial_probe"

python scripts/extract_spatial_goal_features.py \
  --config scripts/config/vlajepa_simple_spatial_goal.yaml \
  --data-root "$DATA_ROOT" \
  --qwen-model "$QWEN_MODEL" \
  --vjepa-checkpoint "$VJEPA21_CKPT" \
  --output "$PROBE_ROOT/features.pt"
```

`--data-root` 与训练入口的 `DATA_ROOT` 相同，指向配置中数据集所在的父目录。`--output` 是特征缓存文件。SONIC 数据改用 `scripts/config/vlajepa_sonic_spatial_goal.yaml`，并使用独立的数据路径和输出目录。

每个样本包含当前空间网格、未来目标网格、当前 Qwen 条件、state、历史网格及其有效性和时间间隔。当前、历史和未来图像分别编码，未来网格只用于监督。Qwen 条件来自当前图像和指令；编码器在 eval 模式下冻结，不参与 head 优化。

默认按 seed 42 选取 12 个训练、4 个验证、4 个测试 episode，每个均匀采样 12 个起点，并要求完整的真实历史和未来帧。可通过 `--train-episodes`、`--val-episodes`、`--test-episodes`、`--anchors-per-episode` 调整规模。SIMPLE 预测未来 30 步（0.6 秒），SONIC 预测未来 40 步（0.8 秒）。

提取器加载 README 中的预训练 Qwen 与 JEPA 权重，新增特殊 token 按生产模型的初始化方式创建；没有加载任务微调后的 VLA checkpoint。state 沿用数据集已有归一化统计，不在这个小划分上重新估计。缓存 manifest 记录权重、源文件、数据元信息的哈希与每个样本的时间位置。

训练、验证、测试必须按完整 episode 分组，同一个 episode 的所有采样起点属于同一划分。对于合并或重编号的数据集，应使用原始 source episode 作为分组单位，避免同一段示范的不同副本进入不同划分。按 episode 留出不等于按采集 session、场景或物体留出，报告结果时应明确实际划分单位。

缓存使用 `schema_version=1`，记录模型结构、各划分的特征、episode ID 和 sample ID。head 脚本检查特征形状、有限值、历史时间方向以及划分之间的 ID 重叠。已有缓存可以供多个 head 实验复用；更换视觉骨干、输入处理或数据划分后，应重新提取。

## 2. 训练空间 head

```bash
python scripts/probe_spatial_goal.py \
  --features "$PROBE_ROOT/features.pt" \
  --output "$PROBE_ROOT/head_seed42" \
  --seed 42 \
  --steps 2000 \
  --batch-size 32 \
  --learning-rate 3e-4
```

脚本默认使用 CUDA，也支持 `--device cpu`。训练时只实例化并优化 `SpatialGoalPredictor`；不加载或更新 JEPA、Qwen、Action Expert、动作 prior、世界模型、空间读取器或动作适配器。

运行包含两个检查：

1. **小样本拟合**：从训练划分选择默认 8 个样本，训练默认 1000 步，检查 head 是否具备拟合这些目标的能力。该检查在同一组小样本上训练和评估，不代表泛化结果。
2. **独立 episode 预测**：在训练划分优化，以验证集的 episode 平均 L1 选择 checkpoint，再对测试划分评估。测试集不参与梯度计算或 checkpoint 选择。

`--overfit-samples`、`--overfit-steps` 控制小样本检查；`--eval-every` 控制验证间隔。主训练默认以 `--history-dropout 0.25` 随机屏蔽有效历史槽位，小样本检查不屏蔽历史。比较超参数时保持特征缓存和划分固定，使用验证集选择设置，再报告测试结果。

扩大数据量时，可锁定已有 episode 的划分，再为各划分补充新的来源：

```bash
python scripts/extract_spatial_goal_features.py \
  --config scripts/config/vlajepa_simple_spatial_goal.yaml \
  --data-root "$DATA_ROOT" \
  --qwen-model "$QWEN_MODEL" \
  --vjepa-checkpoint "$VJEPA21_CKPT" \
  --split-manifest "$PROBE_ROOT/features.manifest.json" \
  --train-episodes 60 --val-episodes 10 --test-episodes 10 \
  --anchors-per-episode 24 \
  --output "$PROBE_ROOT/extended/features.pt"

python scripts/probe_spatial_goal.py \
  --features "$PROBE_ROOT/extended/features.pt" \
  --output "$PROBE_ROOT/extended/head_seed42" \
  --steps 20000 --overfit-steps 10000 --eval-every 500
```

`--split-manifest` 保留已有来源的归属，包括留出的测试 episode；新增来源按 seed 分配。增大采样数会在每个 episode 中重新均匀选择起点。已有缓存和 manifest 不会被覆盖。小规模检查用于调试；正式实验需要更大数据量、充分训练及预先固定的评估方案。多次查看过的测试集不能当作最终论文中从未使用过的留出集。

已有特征可以随数据规模扩大而复用。例如，数据集共 99 个 episode，已有的 60/10/10 划分可以扩成 79/10/10：在提取命令中设置 `--train-episodes 79 --val-episodes 10 --test-episodes 10`，加入 `--split-manifest "$PROBE_ROOT/extended/features.manifest.json" --reuse-features "$PROBE_ROOT/extended/features.pt"`，并指定新的输出路径。保持每个 episode 的采样数不变时，只编码新增来源中的样本。

复用前检查编码配置、权重、源代码、数据元信息、指令与张量格式；同一样本必须保持原划分及时间位置。缓存应来自同一份未修改的数据集；更换视频或观测内容后重新提取。缓存完成校验后才发布最终文件，便于训练进程接续读取。

## 3. 比较预测器与基线

所有方法在相同测试样本和冻结目标空间中评估：

| 方法 | 预测方式 | 比较目的 |
|---|---|---|
| `copy_current` | 将当前网格直接作为未来预测 | 检查模型是否超过“画面保持不变” |
| `training_median_delta` | 当前网格加上训练集中逐坐标的未来变化中位数 | 检查模型是否超过固定的典型变化 |
| `spatial_predictor` | 根据当前网格、Qwen 条件、state 和历史预测未来网格 | 测量训练后的空间 head |
| `predictor_no_history` | 对同一个已训练 head 屏蔽历史 | 检查模型对历史输入的依赖 |

`training_median_delta` 的统计量只从训练集计算。`predictor_no_history` 是同一模型的输入干预；若要测量“训练时引入历史”的贡献，需要另行训练无历史模型。

指标包括：

| 指标 | 定义 |
|---|---|
| `l1` | 所有网格 token 的原始特征绝对误差均值 |
| `cosine_error` | 各 token 的 `1 - cosine_similarity` 均值 |
| `top_change_l1` | 当前到真实未来变化最大的 25% 网格位置上的 L1 |

变化位置根据真实未来与当前特征确定，只用于评估；所有方法使用同一组位置。这些位置可能反映相机运动、背景变化或物体运动，不能直接称为目标物体区域或抓取区域。

报告同时提供样本平均、episode 平均和每个 episode 的结果。checkpoint 按 episode 平均 L1 选择，使长轨迹不会仅因采样点更多而获得更大的验证权重。先比较完整网格误差，再检查变化区域和各 episode 的一致性：仅在静态背景上取得低误差，不能说明模型学会了未来变化。

## 4. 输出与使用范围

每次实验使用一个新的输出目录，生成：

| 文件 | 内容 |
|---|---|
| `report.json` | 配置、特征元数据、最佳验证步数、基线及预测指标 |
| `overfit_curve.jsonl` | 小样本拟合曲线 |
| `fit_curve.jsonl` | 主训练与验证曲线 |
| `prediction_probe.png` | 拟合曲线和测试基线比较图 |
| `spatial_goal_head.pt` | 验证集选出的空间 head 参数及结构信息 |
| `test_predictions.pt` | 各方法对测试样本的预测 |

将特征缓存、head 权重和完整预测张量保存在仓库之外的实验目录，发布时使用单独的数据或模型托管位置，不提交到 Git。`spatial_goal_head.pt` 只包含空间预测器，不是可直接启动完整 policy 的 checkpoint。

这个实验回答的是“固定预训练特征下，空间 head 能否学习并泛化未来网格预测”。较低的预测误差不等于抓取成功，也没有单独验证任务区域读取、动作解码或候选排序；这些模块的作用通过完整 policy 的消融和闭环任务实验评估。
