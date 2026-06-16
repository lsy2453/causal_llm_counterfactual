# PSAHealth Contrastive CBO 实验结果解读与方法修正建议

**报告类型**：实验结果解读与方法改进建议  
**实验对象**：PSAHealth benchmark  
**方法名称**：Contrastive CBO / LLM-guided feasible contrast selection  
**实验选择器**：Kimi selector  
**实验轮数**：3 trials  
**生成日期**：2026-06-15  

---

## 1. 实验结果摘要

本次实验在 PSAHealth benchmark 上运行 contrastive CBO，最终结果为：

```json
{
  "benchmark": "PSAHealth",
  "method": "contrastive",
  "best_y": 2.497938729865327,
  "best_x": [0.875, 1.0],
  "best_intervention_set": "do_aspirin_statin",
  "simple_regret": 0.024062499999999876,
  "extra_metrics": {
    "global_optimum_y_grid": 2.473876229865327,
    "global_optimum_set_grid": "do_aspirin_statin",
    "global_optimum_x_grid": [1.0, 1.0],
    "mechanism_coverage": 1.0,
    "explanation_fidelity_proxy": 1.0,
    "num_contrast_results": 12.0
  }
}
```

从最终结果看，方法在 3 次干预预算内找到：

$$
\hat{x}=(\text{Aspirin}=0.875,\text{Statin}=1.0)
$$

对应目标值为：

$$
\hat{y}=2.497938729865327
$$

网格全局最优为：

$$
x^\star=(\text{Aspirin}=1.0,\text{Statin}=1.0)
$$

$$
y^\star=2.473876229865327
$$

因此 simple regret 为：

$$
r_T=\hat{y}-y^\star=0.0240625
$$

相对最优值的误差约为：

$$
\frac{0.0240625}{2.4738762}\approx 0.97\%
$$

因此，该实验最终性能较好。方法没有精确命中 $$(1.0,1.0)$$，但成功找到接近全局最优的联合干预区域。

---

## 2. 任务方向判断：PSAHealth 是最小化任务

由于 simple regret 由：

$$
2.4979387-2.4738762=0.0240625
$$

得到，说明该任务使用的是最小化目标，即：

$$
\min_x g(x)
$$

因此，目标值越低越好。

这一点非常重要，因为当前日志中使用了：

```text
P(delta>0)
```

但如果定义：

$$
\Delta=g(a)-g(b)
$$

则在最小化任务中，候选 $$a$$ 优于对照 $$b$$ 的条件是：

$$
\Delta<0
$$

而不是：

$$
\Delta>0
$$

因此，日志中的 $$P(\Delta>0)$$ 容易误导实验解释。建议后续将其统一改为方向敏感的：

$$
P_{\text{better}}(a,b)=
\begin{cases}
P(\Delta>0), & \text{maximization task}\\
P(\Delta<0), & \text{minimization task}
\end{cases}
$$

并在日志中输出：

```text
P(candidate_better)
```

而不是：

```text
P(delta>0)
```

---

## 3. best_found_curve 解读

本次 best-found curve 为：

```json
[
  2.6182512298653267,
  2.6182512298653267,
  2.497938729865327
]
```

由于任务是最小化，因此曲线越低越好。

三轮结果可解释为：

| Step | 被选择干预 | 目标值 $g(x)$ | 是否改善 |
|---|---|---:|---|
| 1 | do(Aspirin, Statin) = [0.25, 1.0] | 2.618251 | 一般 |
| 2 | do(Statin) = [1.0] | 2.630084 | 未改善 |
| 3 | do(Aspirin, Statin) = [0.875, 1.0] | 2.497939 | 明显改善 |

该曲线说明：前两轮没有取得实质改进，第三轮成功跳到接近全局最优的联合干预区域。

需要注意的是，日志中出现：

```text
[Current best observed do-outcome] 2.591206
```

但 Step 1 后的 best-found 记录为 2.618251。由于 2.591206 小于 2.618251，如果当前已有最优点被纳入统计，则 Step 1 的 best-found 不应变为 2.618251。

因此，当前 best_found_curve 很可能只统计了本次 sequential trials 中新评估的点，而没有纳入初始数据中的 current best。建议后续将指标拆分为：

```text
best_found_curve_new_trials_only
best_found_curve_including_initial_data
```

否则容易造成实验解读歧义。

---

## 4. Step 1 解读

### 4.1 Base CBO/EI 候选

Step 1 中 base CBO/EI 给出：

```text
do_aspirin EI=0.000065
do_statin EI=0.007749
do_aspirin_statin x=[0.2500, 1.0000] EI=0.058855
```

near-tie pool 只有一个候选：

```text
candidate=do_aspirin_statin x=[0.2500, 1.0000]
```

因此 Step 1 中 contrastive layer 实际上没有候选间重排序空间，只能选择：

$$
x=(0.25,1.0)
$$

对应真实 do-evaluation 为：

$$
g(x)=2.618251
$$

该值不算最优，但干预集合选择正确，即选择了联合干预 $$\text{do}(\text{Aspirin},\text{Statin})$$。

### 4.2 对照选择

Step 1 的 contrast pool 中有 6 个有效对照，其中 Kimi 选择了 3 个 historical comparison：

$$
[0.5436,0.9351]
$$

$$
[0.6066,0.7295]
$$

$$
[0.8159,0.0027]
$$

其中第三个对照得到：

```text
mean_delta = -0.273151
P(delta>0)=0.1176
disc=1.187170
CFScore=1.195111
```

由于任务是最小化，$$\Delta<0$$ 表示候选优于对照。因此这个结果说明候选 $$[0.25,1.0]$$ 明显优于历史低 Statin 点 $$[0.8159,0.0027]$$。

然而，这个对照可能过于容易，因为 $$[0.8159,0.0027]$$ 的 Statin 几乎为零，可能本身就是较差区域。因此它虽然能提供较大可区分性，但未必是最有决策价值的对照。

---

## 5. Step 2 解读：暴露出 CFScore 的核心问题

Step 2 是本次实验最关键的部分。

Base CBO/EI 给出：

```text
do_aspirin_statin x=[0.8750, 1.0000] EI=0.026594
do_statin x=[1.0000] EI=0.007749
```

由于：

$$
0.026594-0.007749=0.018845<0.02
$$

二者均进入 near-tie pool。

从真实结果看：

$$
g(\text{do\_aspirin\_statin}=[0.875,1.0])=2.497939
$$

$$
g(\text{do\_statin}=[1.0])=2.630084
$$

因此联合干预候选明显更优。

但是 contrastive layer 在 Step 2 选择了：

$$
\text{do}(\text{Statin}=1.0)
$$

原因是该候选的 CFScore 极高：

```text
do_statin score=5.101049
```

这个高分来自与历史低 Statin 点的对照：

```text
x_b=[0.0165]
mean_delta=-0.440029
P(delta>0)=0.0000
disc=5.099933
CFScore=5.101049
```

由于任务为最小化，$$\Delta=-0.440029<0$$ 说明 do(Statin=1.0) 明显优于 do(Statin=0.0165)。但这并不说明 do(Statin=1.0) 比联合干预 $$[0.875,1.0]$$ 更值得实验。

这暴露出当前 CFScore 的核心缺陷：

> 当前 CFScore 会偏好“相对于一个很差历史点差异很大”的候选，而不一定偏好“相对于有竞争力对照具有机制优势”的候选。

换言之，当前评分容易奖励 easy contrast，而不是 hard contrast。

---

## 6. Step 3 解读

Step 3 中 base CBO/EI 再次给出：

```text
do_aspirin_statin x=[0.8750, 1.0000] EI=0.026594
```

near-tie pool 只有该候选，因此 contrastive layer 选择：

$$
x=(0.875,1.0)
$$

真实 do-evaluation 为：

$$
g(x)=2.497939
$$

该结果成为最终 best-found。它与全局最优：

$$
x^\star=(1.0,1.0)
$$

非常接近。

Step 3 的 contrast pool 中其实包含一个非常重要的 local contrast：

$$
b=(1.0,1.0)
$$

该点正是网格全局最优。但 Kimi selector 仍然主要选择 historical comparison，而没有选择该 single-variable increase contrast。这说明当前 LLM/selector 的 prompt 或选择规则存在偏差：它过度偏好历史对照，而没有充分利用局部机制对照。

---

## 7. 当前结果的积极信号

### 7.1 最终识别了正确干预集合

最终选择：

```text
do_aspirin_statin
```

而网格全局最优也属于：

```text
do_aspirin_statin
```

这说明方法最终识别出了正确的干预集合。

### 7.2 最终点接近全局最优

最终点：

$$
(0.875,1.0)
$$

全局最优：

$$
(1.0,1.0)
$$

simple regret：

$$
0.0240625
$$

说明方法在非常有限的 3 次 trial 下已经接近最优区域。

### 7.3 contrastive layer 没有破坏最终方向

尽管 Step 2 选择了较差的单独 Statin 干预，但 Step 3 回到了联合干预并取得较优结果。因此，当前 near-tie 设计在一定程度上限制了 contrastive layer 的负面影响。

### 7.4 机制覆盖与解释忠实代理指标为 1.0

结果中：

```json
"mechanism_coverage": 1.0,
"explanation_fidelity_proxy": 1.0
```

说明当前实现中 contrastive 模块覆盖了预设机制，并且解释没有偏离结构化结果。不过这两个指标目前可能较宽松，后续应进一步细化。

---

## 8. 当前结果暴露的问题

### 8.1 CFScore 奖励 easy contrast

当前 CFScore 主要由 discriminability 主导：

$$
Disc=\frac{|\mu_\Delta|}{\sigma_\Delta}
$$

因此，只要候选与某个对照差异很大，就能获得高分。问题是：差异大的对照可能是明显很差的历史点，并不具备真正的决策价值。

### 8.2 概率方向不适配最小化任务

日志中报告 $$P(\Delta>0)$$，但 PSAHealth 是最小化任务。应将其改为方向敏感的：

$$
P_{\text{better}}
$$

### 8.3 LLM selector 过度偏好 historical comparison

三轮中 Kimi selector 主要选择 historical comparison，而没有充分选择：

- single-variable decrease；
- single-variable increase；
- local contrast；
- dose-reduction contrast；
- mechanism-path contrast；
- hard near-tie contrast。

这会削弱方法中“LLM 选择机制有意义对照”的学术贡献。

### 8.4 未充分利用全局最优附近 local contrast

Step 3 中候选池包含：

$$
b=(1.0,1.0)
$$

这是非常关键的对照，因为它能回答：

> 是否应该将 Aspirin 从 0.875 提高到 1.0？

但 selector 没有选择它。该现象表明 selector 需要加入强制 local contrast 或 hard contrast 规则。

### 8.5 best_found_curve 统计口径不清

日志中的 initial current best 和 best-found curve 存在口径不一致问题。建议明确是否包含初始数据。

---

## 9. 关键修正建议

### 9.1 修正 \(P(\Delta>0)\) 为 \(P_{	ext{better}}\)

定义：

$$
\Delta=g(a)-g(b)
$$

对于最大化任务：

$$
P_{\text{better}}=P(\Delta>0)
$$

对于最小化任务：

$$
P_{\text{better}}=P(\Delta<0)
$$

日志中统一输出：

```text
P(candidate_better)
```

该修正可以避免解释方向错误。

---

### 9.2 使用方向敏感且对照质量加权的 CFScore

建议将当前 CFScore 修正为：

$$
CFScore(a,b)
=
w_{\text{rel}}(a,b)
\cdot
w_{\text{hard}}(a,b)
\cdot
P_{\text{better}}(a,b)
\cdot
\frac{|\mu_\Delta|}
{\sigma_\Delta+\eta}
$$

其中：

- $$w_{\text{rel}}(a,b)$$：语义相关性权重；
- $$w_{\text{hard}}(a,b)$$：hard contrast 权重；
- $$P_{\text{better}}(a,b)$$：候选优于对照的概率；
- $$\frac{|\mu_\Delta|}{\sigma_\Delta+\eta}$$：后验可区分性；
- $$\eta$$：数值稳定项。

对于明显很差、很远、无竞争力的历史点，应令：

$$
w_{\text{hard}}\ll 1
$$

从而避免“打败很差点”导致 CFScore 虚高。

---

### 9.3 定义 hard contrast 权重

可定义：

$$
w_{\text{hard}}(a,b)
=
w_y(a,b)\cdot w_x(a,b)\cdot w_\alpha(a,b)
$$

其中：

#### 后验均值接近性

$$
w_y(a,b)
=
\exp
\left(
-
\frac{|\mu_t(a)-\mu_t(b)|}{\tau_y}
\right)
$$

#### 输入距离接近性

$$
w_x(a,b)
=
\exp
\left(
-
\frac{d(a,b)}{\tau_x}
\right)
$$

#### Acquisition 接近性

$$
w_\alpha(a,b)
=
\exp
\left(
-
\frac{|\alpha_t(a)-\alpha_t(b)|}{\tau_\alpha}
\right)
$$

这样可以鼓励选择与候选 $$a$$ 真正具有竞争关系的对照，而不是明显较差的历史点。

---

### 9.4 限制 historical comparison 数量

建议设置：

```text
max_historical_contrasts = 1
min_local_contrasts = 1
min_same_intervention_set_contrasts = 1
```

这样可以强制 selector 选择至少一个局部对照和一个同干预集合对照。

例如对候选：

$$
a=(0.875,1.0)
$$

应优先包含：

$$
b_1=(1.0,1.0)
$$

$$
b_2=(0.725,1.0)
$$

$$
b_3=(0.875,0.85)
$$

这些对照比远距离历史点更能回答机制问题。

---

### 9.5 加入对照选择约束

建议在 prompt 或 selector 中加入：

```json
{
  "selection_constraints": {
    "must_select_one_local_single_variable_contrast": true,
    "must_select_one_same_intervention_set_contrast": true,
    "must_select_at_most_one_historical_contrast": true,
    "prefer_hard_contrasts_with_similar_posterior_mean": true,
    "avoid_trivial_low_performing_historical_contrasts": true
  }
}
```

这可以缓解 Kimi selector 当前过度选择 historical comparison 的问题。

---

### 9.6 将 best_found_curve 分为两类

建议输出：

```json
{
  "best_found_curve_new_trials_only": [...],
  "best_found_curve_including_initial_data": [...]
}
```

其中：

- `new_trials_only` 用于衡量新实验点内部的改进；
- `including_initial_data` 用于完整 BO/CBO 性能曲线。

---

## 10. 改进后的实验日志建议

建议将每个 contrast result 输出为：

```json
{
  "contrast_type": "local_single_variable_increase",
  "x_b": [1.0, 1.0],
  "objective_direction": "minimize",
  "mean_delta": 0.0241,
  "var_delta": 0.0123,
  "p_candidate_better": 0.41,
  "disc": 0.22,
  "hardness_weight": 0.95,
  "semantic_relevance_weight": 1.00,
  "cfscore": 0.086
}
```

候选级别输出：

```json
{
  "candidate": [0.875, 1.0],
  "base_EI": 0.026594,
  "selected_contrasts": [
    "local_single_variable_increase",
    "local_single_variable_decrease",
    "historical_comparison"
  ],
  "candidate_cfscore": 0.494,
  "near_tie_gap": 0.000000
}
```

这样更容易支撑论文中的实验分析。

---

## 11. 可用于论文的结果表述

### 11.1 正面结果表述

英文：

> On the PSAHealth benchmark, the proposed contrastive CBO method identified the correct optimal intervention family, namely the joint intervention on Aspirin and Statin. Within three intervention trials, it reached \(x=(0.875,1.0)\), achieving an outcome of \(2.4979\), close to the grid optimum \(2.4739\) at \(x=(1.0,1.0)\), with a simple regret of \(0.0241\). This suggests that the contrastive decision layer can guide CBO toward the correct joint intervention region under a limited intervention budget.

中文：

> 在 PSAHealth benchmark 上，本文方法最终识别出了正确的最优干预集合，即 Aspirin 与 Statin 的联合干预。在 3 次干预预算内，方法达到 \(x=(0.875,1.0)\)，目标值为 \(2.4979\)，接近网格全局最优 \(x=(1.0,1.0)\) 处的 \(2.4739\)，simple regret 为 \(0.0241\)。这说明 contrastive decision layer 能够在有限预算下将 CBO 引导至正确的联合干预区域。

### 11.2 局限性表述

英文：

> The intermediate trajectory also reveals a failure mode of the first-version CFScore. In Step 2, the contrastive layer selected a marginal statin intervention because it was highly distinguishable from a poor low-statin historical contrast, even though the joint aspirin-statin candidate yielded a better eventual outcome. This indicates that discriminability alone may over-reward easy historical contrasts and motivates a direction-aware, hard-contrast-weighted CFScore.

中文：

> 中间轨迹也暴露了第一版 CFScore 的一个失败模式。在第 2 步中，contrastive layer 选择了单独 Statin 干预，因为它相对于一个低 Statin 的较差历史对照具有很高的可区分性；然而实际结果表明，联合 Aspirin-Statin 候选具有更好的目标值。这说明仅依赖可区分性可能会过度奖励 easy historical contrast，因此需要引入方向敏感的 \(P_{	ext{better}}\) 和 hard-contrast 加权机制。

---

## 12. 总体结论

本次 PSAHealth 实验可以得出以下结论：

1. **最终优化性能较好**  
   方法在 3 次 trial 后找到 \((0.875,1.0)\)，距离网格最优 \((1.0,1.0)\) 很近，simple regret 仅为 \(0.0241\)。

2. **干预集合识别正确**  
   最终选择的干预集合是 `do_aspirin_statin`，与网格全局最优干预集合一致。

3. **contrastive layer 具有潜力**  
   方法能够构造并评估多个反事实/干预对照，并在最终阶段帮助定位接近最优的联合干预区域。

4. **第一版 CFScore 存在明显风险**  
   当前 CFScore 会奖励“相对于很差历史点的巨大差异”，从而可能在 near-tie pool 中误选较差候选。

5. **LLM selector 需要加强约束**  
   当前 Kimi selector 过度偏好 historical comparison，未充分选择 local single-variable contrast 和 hard near-tie contrast。

6. **下一步应重点修正评分机制**  
   建议引入方向敏感的 \(P_{	ext{better}}\)、hard contrast weighting、historical contrast 数量限制以及 local contrast 强制选择。

一句话总结：

> 该实验初步证明 contrastive CBO 能在 PSAHealth 上逼近最优联合干预，但也清楚暴露出第一版 contrastive reranking 的核心缺陷：CFScore 需要从“可区分性最大”升级为“有竞争力、方向正确且机制相关的对照价值最大”。

---

## 13. 建议的下一版方法公式

推荐下一版使用：

$$
CFScore(a,b)
=
w_{\text{rel}}(a,b)
\cdot
w_{\text{hard}}(a,b)
\cdot
P_{\text{better}}(a,b)
\cdot
\frac{|\mu_\Delta|}
{\sigma_\Delta+\eta}
$$

候选级聚合：

$$
CFScore(a)
=
\max_{b\in B_{LLM}(a)}
CFScore(a,b)
$$

或更稳健地：

$$
CFScore(a)
=
\frac{1}{|B_{LLM}(a)|}
\sum_{b\in B_{LLM}(a)}
CFScore(a,b)
$$

同时要求 \(B_{LLM}(a)\) 至少包含：

- 一个 local single-variable contrast；
- 一个 same-intervention-set contrast；
- 至多一个 historical comparison；
- 不选择明显低质量或过远的 easy contrast。

---

## 14. 推荐的下一步实验

下一步建议进行以下实验：

1. **原始 CFScore vs hard-contrast-weighted CFScore**  
   检验修正后是否避免 Step 2 的误选。

2. **Kimi selector vs rule-constrained selector**  
   检验选择约束是否提升 local contrast 和 hard contrast 的比例。

3. **with vs without historical contrast cap**  
   检验 historical comparison 是否过度主导。

4. **不同 near-tie delta**  
   例如：
   $$
   \delta\in\{0.005,0.01,0.02,0.05\}
   $$

5. **报告 contrast type distribution**  
   例如每轮选中的 contrast 中：
   - historical comparison 占比；
   - local contrast 占比；
   - same-intervention-set contrast 占比；
   - mechanism-path contrast 占比。

6. **加入 best_found_curve_including_initial_data**  
   避免 BO/CBO 性能曲线解释歧义。

---

## 15. 最终建议

本次结果不应简单解读为“方法已经完全有效”，也不应解读为“方法失败”。更准确的判断是：

> 方法在最终性能上显示出潜力，但第一版 CFScore 的设计还不够稳健。它能够帮助找到正确联合干预区域，但也会被 easy historical contrast 误导。下一版应将对照选择从“可区分性最大”升级为“hard、local、方向正确、机制相关”的对照价值最大。

这正好可以成为论文方法发展的重要论证线索：第一版实验暴露问题，改进版通过 hard-contrast-weighted CFScore 和 constrained selector 解决该问题，从而增强方法的学术严谨性和实证说服力。
