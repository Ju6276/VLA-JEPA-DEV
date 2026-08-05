# One-Stage Action-Grounded VLA-JEPA：简版逻辑

当前版本是一阶段联合训练，不再拆成“先训 world model、再训 policy”两个 stage。

一句话故事：训练时用未来端点构造 privileged latent teacher，部署侧 student 只根据当前观测预测同一 latent action；再用完整动作序列重建把 latent transition 落到机器人控制语义上，同时用 Knowledge Insulation 阻断 flow loss 经 dynamics tokens 直接回写 latent 分支。

## 1. 三条信息路径

### Student（训练和部署都存在）

```text
current image + language
        -> Qwen3-VL
        -> current_context
        -> StudentCurrentAdapter
        -> u_student

current_context + state_0
        -> StudentPredictor
        -> z_student

u_student + z_student
        -> SharedWorldDecoder
        -> u_student_hat_T
```

Student 不读取未来帧或 `state_T`，因此部署时是因果可用的。

### Teacher（只在训练时存在）

```text
video -> frozen V-JEPA -> u_0_target, u_T_target

u_0_target + u_T_target + state_0 + state_T
        -> TeacherEncoder (LaWAM LAMEncoder)
        -> z_teacher

u_0_target + z_teacher
        -> SharedWorldDecoder
        -> u_teacher_hat_T
```

旧实现只把当前视觉端点传给 Teacher；当前版本已经修复为当前、未来两个视觉端点都进入 Teacher。

### Action grounding（只在训练时增加监督）

```text
delta_gt   = u_T_target - u_0_target
delta_pred = u_student_hat_T - u_student

delta_gt   -> MultiStepDeltaActionDecoder -> complete action chunk
delta_pred -> MultiStepDeltaActionDecoder -> complete action chunk
```

这部分借鉴 Delta-JEPA 的 long-horizon displacement-to-action 思路。它不是第二阶段，也不是额外部署模块。

## 2. Knowledge Insulation

主动作路径是：

```text
embodied tokens
+ stopgrad(u_student_hat_T)
+ stopgrad(z_student)
        -> projection
        -> FlowMatching Action Head
        -> action
```

因此：

- Action Head 和两个投影层仍可由 `L_act` 训练；
- `L_act` 不会穿过 latent interface 改写 StudentPredictor / SharedWorldDecoder；
- dynamics 仍会收到 `L_current_align`、`L_student_wm`、`L_distill` 和 `L_ldad_pred` 的梯度。

这就是第一版采用的 LaWAM-style Knowledge Insulation。

## 3. 一阶段总损失

```text
L_total =
    lambda_act        * L_act
  + lambda_current    * L_current_align
  + lambda_student_wm * L_student_wm
  + lambda_teacher_wm * L_teacher_wm
  + lambda_distill    * L_distill
  + lambda_ldad_gt    * L_ldad_gt
  + lambda_ldad_pred  * L_ldad_pred
  + lambda_latent     * L_latent
  + lambda_state      * L_state
```

默认关闭 `L_latent` 和 `L_state`，避免第一版同时堆叠过多假设。当前也没有加入 Music-JEPA temporal prior、Causal-JEPA object masking、verifier 或显式外部 subgoal proposal。

## 4. 三个工作的结合点

| 来源 | 当前代码真正采用的部分 | 位置 |
|---|---|---|
| VLA-JEPA base | Qwen student、冻结 V-JEPA target、future latent alignment、Flow Action Head | `VLA_JEPA.py` |
| LaWAM | `LAMEncoder`/`LAMDecoder_v2`、privileged latent action、teacher→student distillation、Knowledge Insulation | `privileged_latent.py`, `VLA_JEPA.py` |
| Delta-JEPA | long-horizon latent displacement、完整 action chunk reconstruction | `delta_action_decoder.py` |

这里没有简单叠加三个网络。统一接口是 transition latent：Teacher 负责训练期识别 transition，Student 负责部署期预测 transition，shared decoder 检验其未来可预测性，Delta decoder 检验其动作可执行性。

## 5. 推理

推理只需要：

```text
image + language + state_0
        -> Student
        -> u_student_hat_T, z_student
        -> Action Head
        -> normalized_actions
```

不会加载或运行 V-JEPA、TeacherEncoder、MultiStepDeltaActionDecoder，也不需要 `video`、`state_T` 或未来信息。

更完整的张量、loss 与配置说明见 `privileged_e2e_scheme.md`。
