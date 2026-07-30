# 训练实例核对提问模板

把下面整段发给另一个 agent。先把 `【实例】` 填成你要核对的启动脚本或 config（二选一或都填），然后让它**只针对这一次实例**回答；不要泛泛介绍仓库。

---

## 发给 Agent 的提问（复制后改【实例】）

```text
请针对下面这一次训练实例，逐条核对并回答。依据仓库里的实际脚本/config/代码，给出具体路径与数值，不要写通用说明。

【实例】
- 启动脚本：scripts/vlajepa_XXXXXXXX.sh
- 或 config：scripts/config/vlajepa_XXXXXXXX.yaml
（以实际会跑的那个为准；若脚本写死了 config，以脚本里的 --config_yaml 为准）

请按编号回答：

1. 训练的数据集是什么？
   - data_root_dir、data_mix、实际数据子目录分别是什么？
   - action_type 是什么？action_dim / state_dim 是多少？
   - 对应的 robot_type / DataConfig 类名是什么？
   - 该数据目录下 modality.json / stats.json 是否存在？

2. Qwen 权重是什么？
   - framework.qwenvl.base_vlm 的完整路径是什么？
   - attn_implementation 用的是什么？
   - 该路径在磁盘上是否存在？

3. JEPA encoder 是什么？
   - framework.vj2_model.base_encoder 的完整路径是什么？
   - arch / image_size / num_frames 分别是什么？
   - 走的是 V-JEPA 2 还是 2.1（.pt → 2.1；HF 目录 → 2）？
   - 该路径在磁盘上是否存在？

4. 训练超参与归一化：
   - 实际用的是哪一个 config 文件？run_id / 输出目录是什么？
   - 数据归一化怎么做的？（各 state/action key 的 normalization_modes：min_max / mean_std / …）
   - per_device_batch_size（单卡 BS）是多少？若知道 GPU 数，全局 BS 约多少？
   - max_train_steps 是多少？save_interval 是多少？
   - 有没有 pretrained_checkpoint？若有，路径是什么？

5. Chunk 与可训练模块：
   - future_action_window_size / past_action_window_size / action_horizon 分别是多少？
   - action chunk（chunk_len）是多少？
   - trainer.freeze_modules 写了什么？
   - 结合代码：训练中哪些模块可训练、哪些实际冻结？（尤其说明 vj_encoder / qwen_vl_interface / action_model / vj_predictor）

要求：
- 每条直接给结论，并标注来自哪个文件的哪个字段。
- 若脚本与 yaml 不一致，以启动命令实际会加载的为准，并指出冲突。
- 发现路径不存在或维度与 DataConfig 不一致时，明确标出「风险」。
```

---

## 使用示例

把【实例】改成例如：

```text
【实例】
- 启动脚本：scripts/vlajepa_all_merged_ft_from70k.sh
```

或：

```text
【实例】
- config：scripts/config/vlajepa_g1_open_faucet_ft.yaml
```

发给 agent 后，用它的回答做开训前核对即可。
