# 严格版本 SCM-GP 数值反事实对照计算报告

**报告主题**：严格 SCM-GP 反事实对照计算  
**适用方法**：语言引导的可行对照选择决策层用于可解释因果贝叶斯优化  
**语言**：中文  
**生成日期**：2026-06-15  

---

## 摘要

本文给出“严格版本”的 SCM-GP 数值反事实对照计算方案。该版本不同于仅比较

\[
g(a)-g(b),\quad g(x)=\mathbb E[Y\mid do(X=x)]
\]

的后验干预对照，而是对同一个已发生事实情境 \(e\) 执行 Pearl 式反事实推理：

\[
\Delta_{a,b}^{CF}(e)
=
Y_a^{CF}(e)-Y_b^{CF}(e)
\]

其中：

\[
Y_a^{CF}(e)=Y_{do(X=a)}(e)
\]

\[
Y_b^{CF}(e)=Y_{do(X=b)}(e)
\]

严格 SCM-GP 版本必须显式完成：

\[
\text{abduction}
\rightarrow
\text{action}
\rightarrow
\text{prediction}
\]

即：

1. 基于事实情境 \(e\) 反推外生噪声；
2. 分别将结构方程替换为候选干预 \(a\) 和对照干预 \(b\)；
3. 在保持同一个外生噪声样本不变的条件下沿因果图前向传播；
4. 得到配对的 \(Y_a^{CF}\) 和 \(Y_b^{CF}\) 样本；
5. 估计反事实差异 \(\Delta_{a,b}^{CF}(e)\) 的后验分布。

该严格版本可作为论文中“强因果语义扩展”或主方法的高级实例化版本，用于回应审稿人关于“是否真正执行了反事实推理”的质疑。

---

## 1. 简化版与严格版的根本区别

### 1.1 简化版：后验干预对照

简化版本通常对整体干预效应函数建模：

\[
g(x)=\mathbb E[Y\mid do(X=x)]
\]

并计算：

\[
\Delta_{a,b}^{IC}=g(a)-g(b)
\]

其中 \(IC\) 表示 interventional contrast。

该对象回答的是：

> 在总体平均或 GP 后验意义下，执行干预 \(a\) 相比执行干预 \(b\) 的目标效果差异是多少？

它不依赖某个具体事实情境 \(e\)，也不需要对外生噪声 \(U\) 执行 abduction。

### 1.2 严格版：个体化 SCM 反事实对照

严格版本计算的是：

\[
\Delta_{a,b}^{CF}(e)
=
Y_{do(X=a)}(e)-Y_{do(X=b)}(e)
\]

它回答的是：

> 对同一个已经发生的事实情境 \(e\)，如果执行候选干预 \(a\) 与执行对照干预 \(b\)，目标变量 \(Y\) 的结果会差多少？

该对象必须保持同一个事实情境下的外生背景不变，因此需要推断：

\[
p(U\mid e,D_t,G)
\]

这一步就是 abduction。

### 1.3 对照表

| 项目 | 后验干预对照 | 严格 SCM-GP 反事实对照 |
|---|---|---|
| 计算对象 | \(g(a)-g(b)\) | \(Y_a^{CF}(e)-Y_b^{CF}(e)\) |
| 语义 | 总体平均干预差异 | 同一事实情境下的反事实差异 |
| 是否依赖事实情境 \(e\) | 不一定 | 必须 |
| 是否需要 abduction | 否 | 是 |
| 是否保持同一外生噪声 | 否 | 是 |
| 是否需要结构方程 | 不一定 | 必须 |
| 是否可称为严格反事实 | 不宜 | 可以 |
| 计算难度 | 低 | 高 |
| 数据需求 | 较低 | 较高 |

---

## 2. SCM-GP 基本设定

设系统由结构因果模型表示：

\[
\mathcal M=
\langle
G,\mathcal V,\mathcal U,\mathcal F,P(\mathcal U)
\rangle
\]

其中：

- \(G\)：有向无环因果图；
- \(\mathcal V=\{V_1,\dots,V_p,Y\}\)：内生变量集合；
- \(\mathcal U=\{U_1,\dots,U_p,U_Y\}\)：外生噪声集合；
- \(\mathcal F=\{f_j\}\)：结构方程集合；
- \(P(\mathcal U)\)：外生噪声分布。

对于每个非根节点或可建模节点 \(V_j\)，设：

\[
V_j=f_j(PA_j)+U_j
\]

其中 \(PA_j\) 是 \(V_j\) 在因果图中的父节点集合。

在 SCM-GP 中，不是只对最终目标函数 \(g(x)\) 建 GP，而是对每个结构机制建 GP：

\[
f_j\sim \mathcal{GP}(m_j,k_j)
\]

因此结构方程为：

\[
V_j=f_j(PA_j)+U_j,
\quad
f_j\sim \mathcal{GP}(m_j,k_j),
\quad
U_j\sim P(U_j)
\]

目标变量也可写为：

\[
Y=f_Y(PA_Y)+U_Y,
\quad
f_Y\sim \mathcal{GP}(m_Y,k_Y)
\]

---

## 3. 目标：计算数值反事实对照

给定：

- 当前事实情境 \(e\)；
- 候选干预 \(a\)；
- 由 LLM 从合法数值对照池中选择出的对照干预 \(b\)；
- 历史数据 \(D_t\)；
- 因果图 \(G\)；
- 每个结构方程的 GP 后验。

严格反事实对照定义为：

\[
\Delta_{a,b}^{CF}(e)
=
Y_a^{CF}(e)-Y_b^{CF}(e)
\]

需要估计其后验分布：

\[
p\left(
\Delta_{a,b}^{CF}(e)
\mid e,D_t,G
\right)
\]

并报告：

\[
\mathbb E[\Delta_{a,b}^{CF}(e)]
\]

\[
\operatorname{Var}[\Delta_{a,b}^{CF}(e)]
\]

\[
P(\Delta_{a,b}^{CF}(e)>0)
\]

以及可信区间，例如：

\[
CI_{95\%}
=
[q_{0.025},q_{0.975}]
\]

其中 \(q_{\gamma}\) 是 Monte Carlo 样本分位数。

---

## 4. 严格 SCM-GP 反事实计算三步

## 4.1 Step 1：Abduction，反推外生噪声

设事实情境为：

\[
e=\{v_j^{obs}:j\in \mathcal O\}
\]

其中 \(\mathcal O\) 是已观测节点集合。

对于每个已观测节点 \(V_j\)，结构方程为：

\[
V_j=f_j(PA_j)+U_j
\]

若无测量误差，则对给定结构函数样本 \(f_j^{(s)}\)，外生噪声样本可由残差得到：

\[
u_j^{(s)}
=
v_j^{obs}
-
f_j^{(s)}(pa_j^{obs})
\]

这里：

- \(f_j^{(s)}\sim p(f_j\mid D_t)\)；
- \(pa_j^{obs}\) 是事实情境中父节点的观测值；
- \(u_j^{(s)}\) 是与该函数样本匹配的外生噪声样本。

重复采样即可得到：

\[
\{U^{(s)}\}_{s=1}^{S}
\sim
p(U\mid e,D_t,G)
\]

### 4.1.1 有测量噪声时的 abduction

若观测模型为：

\[
V_j^{obs}=f_j(PA_j)+U_j+\epsilon_j
\]

其中：

\[
U_j\sim \mathcal N(0,\tau_j^2)
\]

\[
\epsilon_j\sim \mathcal N(0,\sigma_j^2)
\]

定义残差：

\[
r_j
=
v_j^{obs}
-
f_j^{(s)}(pa_j^{obs})
\]

则：

\[
r_j=U_j+\epsilon_j
\]

由高斯条件分布可得：

\[
U_j\mid r_j
\sim
\mathcal N
\left(
\frac{\tau_j^2}{\tau_j^2+\sigma_j^2}r_j,
\frac{\tau_j^2\sigma_j^2}{\tau_j^2+\sigma_j^2}
\right)
\]

因此可从该条件分布中采样：

\[
u_j^{(s)}
\sim
p(U_j\mid r_j)
\]

该处理能避免把全部残差都归因于外生噪声，从而在有测量误差时更稳健。

---

## 4.2 Step 2：Action，替换结构方程

对于候选干预 \(a\)，执行：

\[
do(X_{\mathcal I}=a)
\]

即将可干预变量集合 \(\mathcal I\) 的结构方程替换为常数：

\[
X_{\mathcal I}:=a
\]

对于对照干预 \(b\)，执行：

\[
do(X_{\mathcal I}=b)
\]

即：

\[
X_{\mathcal I}:=b
\]

注意：

- 只替换可干预变量的结构方程；
- 不应替换中介变量或目标变量；
- 不应修改不可干预变量；
- 候选世界和对照世界必须使用同一个 \(U^{(s)}\)，否则不是同一事实情境下的反事实比较。

---

## 4.3 Step 3：Prediction，沿因果图前向传播

在同一个函数样本 \(f^{(s)}\) 和同一个外生噪声样本 \(U^{(s)}\) 下，分别构造两个世界。

### 候选世界 \(a\)

若 \(V_j\in X_{\mathcal I}\)，则：

\[
V_j^{a,(s)}=a_j
\]

若 \(V_j\notin X_{\mathcal I}\)，则按因果图拓扑顺序计算：

\[
V_j^{a,(s)}
=
f_j^{(s)}(PA_j^{a,(s)})
+
u_j^{(s)}
\]

最终得到：

\[
Y_a^{CF,(s)}
\]

### 对照世界 \(b\)

若 \(V_j\in X_{\mathcal I}\)，则：

\[
V_j^{b,(s)}=b_j
\]

若 \(V_j\notin X_{\mathcal I}\)，则：

\[
V_j^{b,(s)}
=
f_j^{(s)}(PA_j^{b,(s)})
+
u_j^{(s)}
\]

最终得到：

\[
Y_b^{CF,(s)}
\]

### 配对反事实差异

对每个 Monte Carlo 样本：

\[
\Delta_{a,b}^{CF,(s)}
=
Y_a^{CF,(s)}
-
Y_b^{CF,(s)}
\]

得到样本集合：

\[
\{\Delta_{a,b}^{CF,(s)}\}_{s=1}^{S}
\]

其均值、方差、胜率和可信区间分别估计为：

\[
\widehat{\mu}_{\Delta}
=
\frac{1}{S}
\sum_{s=1}^{S}
\Delta_{a,b}^{CF,(s)}
\]

\[
\widehat{\sigma}_{\Delta}^{2}
=
\frac{1}{S-1}
\sum_{s=1}^{S}
\left(
\Delta_{a,b}^{CF,(s)}
-
\widehat{\mu}_{\Delta}
\right)^2
\]

\[
\widehat{P}(\Delta>0)
=
\frac{1}{S}
\sum_{s=1}^{S}
\mathbf 1
[
\Delta_{a,b}^{CF,(s)}>0
]
\]

\[
CI_{95\%}
=
[
\operatorname{Quantile}_{0.025}(\Delta),
\operatorname{Quantile}_{0.975}(\Delta)
]
\]

---

## 5. 数值型实验变量完整例子

考虑材料实验中的四个变量：

| 变量 | 含义 | 类型 |
|---|---|---|
| \(T\) | 温度 | 可干预 |
| \(C\) | 催化剂比例 | 可干预 |
| \(M\) | 中间相指标 | 中介变量 |
| \(Y\) | 目标产率 | 目标变量 |

因果图为：

\[
T\rightarrow M\rightarrow Y
\]

\[
C\rightarrow M\rightarrow Y
\]

同时 \(T\) 和 \(C\) 也可直接影响 \(Y\)：

\[
T\rightarrow Y
\]

\[
C\rightarrow Y
\]

结构方程为：

\[
M=f_M(T,C)+U_M
\]

\[
Y=f_Y(M,T,C)+U_Y
\]

对两个机制函数建 GP：

\[
f_M\sim \mathcal{GP}(m_M,k_M)
\]

\[
f_Y\sim \mathcal{GP}(m_Y,k_Y)
\]

---

## 5.1 当前事实情境

假设当前已经执行过候选干预：

\[
a=(T=750,\ C=0.15)
\]

并观测到：

\[
M^{obs}=4.80
\]

\[
Y^{obs}=9.10
\]

LLM 从合法数值对照池中选择一个对照：

\[
b=(T=700,\ C=0.15)
\]

该对照含义为：

> 保持催化剂比例 \(C=0.15\) 不变，仅将温度从 \(750^\circ C\) 降至 \(700^\circ C\)，检验当前高温干预是否具有必要性。

---

## 5.2 GP 后验摘要

假设当前 GP 后验给出如下数值。

中介机制：

\[
f_M(750,0.15)\mid D_t
\sim
\mathcal N(4.50,0.20^2)
\]

\[
f_M(700,0.15)\mid D_t
\sim
\mathcal N(4.20,0.25^2)
\]

二者相关系数为：

\[
\rho_M=0.70
\]

因此协方差为：

\[
\operatorname{Cov}
=
\rho_M\times 0.20\times 0.25
=
0.035
\]

结果机制在事实输入处：

\[
f_Y(4.80,750,0.15)\mid D_t
\sim
\mathcal N(8.70,0.25^2)
\]

在对照输入附近，经 Monte Carlo 传播可近似为：

\[
f_Y(M_b,700,0.15)\mid D_t
\approx
\mathcal N(8.05,0.30^2)
\]

---

## 5.3 Abduction：反推事实外生噪声

### 中介机制噪声

无测量误差近似下：

\[
U_M
=
M^{obs}
-
f_M(750,0.15)
\]

因此：

\[
\mathbb E[U_M]
=
4.80-4.50
=
0.30
\]

由于 \(f_M(750,0.15)\) 的后验标准差为 \(0.20\)，可近似：

\[
U_M
\approx
\mathcal N(0.30,0.20^2)
\]

### 结果机制噪声

\[
U_Y
=
Y^{obs}
-
f_Y(M^{obs},750,0.15)
\]

因此：

\[
\mathbb E[U_Y]
=
9.10-8.70
=
0.40
\]

并可近似：

\[
U_Y
\approx
\mathcal N(0.40,0.25^2)
\]

解释上，这说明该事实样本相对于当前 GP 机制均值存在正向外生偏移。在严格反事实比较中，该偏移必须在候选世界和对照世界中保持一致。

---

## 5.4 Action：构造两个反事实世界

### 候选世界

\[
do(T=750,C=0.15)
\]

### 对照世界

\[
do(T=700,C=0.15)
\]

两个世界使用同一个外生状态：

\[
(U_M,U_Y)
\]

---

## 5.5 Prediction：候选世界

在候选世界中：

\[
M_a^{CF}
=
f_M(750,0.15)+U_M
\]

由于：

\[
U_M=M^{obs}-f_M(750,0.15)
\]

所以：

\[
M_a^{CF}=M^{obs}=4.80
\]

进一步：

\[
Y_a^{CF}
=
f_Y(M_a^{CF},750,0.15)+U_Y
\]

同理，在无测量噪声和完全事实重构情形下：

\[
Y_a^{CF}
\approx
Y^{obs}
=
9.10
\]

这符合反事实推理的 consistency 要求：若反事实干预与事实干预相同，则应重构事实结果。

---

## 5.6 Prediction：对照世界

在对照世界中：

\[
M_b^{CF}
=
f_M(700,0.15)+U_M
\]

将：

\[
U_M=M^{obs}-f_M(750,0.15)
\]

代入得：

\[
M_b^{CF}
=
M^{obs}
+
f_M(700,0.15)
-
f_M(750,0.15)
\]

其均值：

\[
\mathbb E[M_b^{CF}]
=
4.80+4.20-4.50
=
4.50
\]

其方差：

\[
\operatorname{Var}(M_b^{CF})
=
\operatorname{Var}(f_M(700,0.15))
+
\operatorname{Var}(f_M(750,0.15))
-
2\operatorname{Cov}
\]

代入数值：

\[
\operatorname{Var}(M_b^{CF})
=
0.25^2+0.20^2-2(0.035)
\]

\[
=
0.0625+0.0400-0.0700
=
0.0325
\]

因此：

\[
M_b^{CF}
\approx
\mathcal N(4.50,0.180^2)
\]

进一步：

\[
Y_b^{CF}
=
f_Y(M_b^{CF},700,0.15)+U_Y
\]

通过 Monte Carlo 传播，假设得到：

\[
Y_b^{CF}
\approx
\mathcal N(8.45,0.30^2)
\]

---

## 5.7 数值反事实对照结果

候选世界结果：

\[
Y_a^{CF}\approx 9.10
\]

对照世界结果：

\[
Y_b^{CF}\approx \mathcal N(8.45,0.30^2)
\]

因此：

\[
\Delta_{a,b}^{CF}
=
Y_a^{CF}
-
Y_b^{CF}
\]

均值：

\[
\mathbb E[\Delta_{a,b}^{CF}]
=
9.10-8.45
=
0.65
\]

标准差：

\[
\sigma_{\Delta}\approx 0.30
\]

95% 可信区间：

\[
0.65\pm 1.96\times0.30
\]

即：

\[
[0.06,\ 1.24]
\]

候选优于对照的概率：

\[
P(\Delta_{a,b}^{CF}>0)
\approx
\Phi
\left(
\frac{0.65}{0.30}
\right)
\approx
0.985
\]

因此可解释为：

> 在当前事实情境下，保持催化剂比例为 \(0.15\) 不变，如果将温度从 \(750^\circ C\) 降至 \(700^\circ C\)，模型预测目标产率将从约 \(9.10\) 降至约 \(8.45\)。候选干预相对于该温度降低对照的反事实优势均值约为 \(0.65\)，95% 可信区间约为 \([0.06,1.24]\)，候选优于对照的后验概率约为 \(0.985\)。

---

## 6. 严格 SCM-GP 算法伪代码

```text
Algorithm: Strict SCM-GP Counterfactual Contrast

Input:
  Causal graph G
  Structural equations {f_j}
  GP posterior p(f_j | D_t) for each f_j
  Observed factual context e
  Candidate intervention a
  Contrast intervention b
  Number of posterior samples S

Output:
  Posterior samples of Δ_CF(a,b | e)

For s = 1,...,S:

  1. Sample structural functions:
       f_j^(s) ~ p(f_j | D_t)

  2. Abduction:
       For each observed node V_j:
          if no measurement noise:
              u_j^(s) = v_j^obs - f_j^(s)(pa_j^obs)
          else:
              r_j = v_j^obs - f_j^(s)(pa_j^obs)
              u_j^(s) ~ p(U_j | r_j)

  3. Action under candidate a:
       Replace equations of intervention variables:
          X_I := a

  4. Prediction under candidate a:
       Traverse G in topological order:
          if V_j is intervened:
              V_j^{a,(s)} = a_j
          else:
              V_j^{a,(s)} =
                  f_j^(s)(PA_j^{a,(s)}) + u_j^(s)
       Store Y_a^{CF,(s)}

  5. Action under contrast b:
       Replace equations of intervention variables:
          X_I := b

  6. Prediction under contrast b:
       Traverse G in topological order:
          if V_j is intervened:
              V_j^{b,(s)} = b_j
          else:
              V_j^{b,(s)} =
                  f_j^(s)(PA_j^{b,(s)}) + u_j^(s)
       Store Y_b^{CF,(s)}

  7. Paired contrast:
       Δ_s = Y_a^{CF,(s)} - Y_b^{CF,(s)}

Return:
  {Δ_s}_{s=1}^S,
  mean({Δ_s}),
  variance({Δ_s}),
  credible_interval({Δ_s}),
  P(Δ_s > 0)
```

---

## 7. Python 风格实现伪代码

```python
def strict_scm_gp_counterfactual_contrast(
    graph,
    gp_posteriors,
    factual_context,
    candidate_a,
    contrast_b,
    intervention_vars,
    observed_nodes,
    S=1000,
):
    delta_samples = []

    topo_order = topological_sort(graph)

    for s in range(S):

        # 1. Sample structural functions from GP posteriors
        sampled_functions = {
            node: gp_posteriors[node].sample_function()
            for node in graph.nodes
            if node not in intervention_vars
        }

        # 2. Abduction: infer exogenous noise
        u = {}
        for node in observed_nodes:
            if node in intervention_vars:
                continue

            parents = graph.parents(node)
            pa_obs = [factual_context[p] for p in parents]
            pred = sampled_functions[node](pa_obs)

            # no-measurement-noise approximation
            u[node] = factual_context[node] - pred

        # 3. Prediction under candidate a
        world_a = {}
        for node in topo_order:
            if node in intervention_vars:
                world_a[node] = candidate_a[node]
            else:
                parents = graph.parents(node)
                pa_values = [world_a[p] for p in parents]
                world_a[node] = sampled_functions[node](pa_values) + u[node]

        y_a = world_a["Y"]

        # 4. Prediction under contrast b
        world_b = {}
        for node in topo_order:
            if node in intervention_vars:
                world_b[node] = contrast_b[node]
            else:
                parents = graph.parents(node)
                pa_values = [world_b[p] for p in parents]
                world_b[node] = sampled_functions[node](pa_values) + u[node]

        y_b = world_b["Y"]

        # 5. Paired counterfactual contrast
        delta_samples.append(y_a - y_b)

    return {
        "delta_samples": delta_samples,
        "delta_mean": mean(delta_samples),
        "delta_variance": variance(delta_samples),
        "credible_interval_95": quantile(delta_samples, [0.025, 0.975]),
        "prob_candidate_better": mean([d > 0 for d in delta_samples]),
    }
```

---

## 8. 用于决策层的严格 CFScore

对于候选 \(a\)，LLM 选择出的有效对照集合为：

\[
B_{LLM}(a)=\{b_1,\dots,b_m\}
\]

对每个 \(b\)，严格 SCM-GP 得到：

\[
\{\Delta_{a,b}^{CF,(s)}\}_{s=1}^{S}
\]

定义：

\[
\mu_{a,b}^{CF}
=
\frac{1}{S}
\sum_{s=1}^{S}
\Delta_{a,b}^{CF,(s)}
\]

\[
\sigma_{a,b}^{CF}
=
\sqrt{
\frac{1}{S-1}
\sum_{s=1}^{S}
(\Delta_{a,b}^{CF,(s)}-\mu_{a,b}^{CF})^2
}
\]

\[
p_{a,b}^{+}
=
\frac{1}{S}
\sum_{s=1}^{S}
\mathbf 1[
\Delta_{a,b}^{CF,(s)}>0
]
\]

可定义严格反事实可区分性：

\[
CFDisc(a,b)
=
\frac{|\mu_{a,b}^{CF}|}
{\sigma_{a,b}^{CF}+\eta}
\]

其中 \(\eta>0\) 是数值稳定项。

若希望候选 \(a\) 相对于对照明显更优，可以定义：

\[
CFScore(a)
=
\max_{b\in B_{LLM}(a)}
\left[
w_b
\cdot
p_{a,b}^{+}
\cdot
CFDisc(a,b)
\right]
\]

其中 \(w_b\) 是对照语义权重，可由对照类型、验证器评分或专家规则给出。

更保守的聚合方式是：

\[
CFScore(a)
=
\frac{1}{|B_{LLM}(a)|}
\sum_{b\in B_{LLM}(a)}
w_b
\cdot
p_{a,b}^{+}
\cdot
CFDisc(a,b)
\]

若候选池仍采用 near-tie 约束，则最终选择：

\[
x_{t+1}
\in
\arg\max_{a\in \mathcal C_t(\delta_t)}
CFScore(a)
\]

其中：

\[
\mathcal C_t(\delta_t)
=
\{x:\alpha_t^{CBO}(x)\ge \alpha_t^\star-\delta_t\}
\]

---

## 9. 严格解释生成模板

LLM 解释不应直接访问原始 GP 或自行判断因果关系，而应读取结构化计算结果：

```json
{
  "candidate": {"T": 750, "C": 0.15},
  "contrast": {"T": 700, "C": 0.15},
  "factual_context": {
    "observed_M": 4.80,
    "observed_Y": 9.10
  },
  "abduced_noise_summary": {
    "U_M_mean": 0.30,
    "U_Y_mean": 0.40
  },
  "counterfactual_results": {
    "Y_candidate_mean": 9.10,
    "Y_contrast_mean": 8.45,
    "delta_mean": 0.65,
    "delta_sd": 0.30,
    "credible_interval_95": [0.06, 1.24],
    "prob_candidate_better": 0.985
  },
  "causal_path": "T -> M -> Y",
  "contrast_type": "temperature_reduction"
}
```

可生成解释：

> 在当前事实情境下，SCM-GP 通过 abduction 推断该样本在中介机制和目标机制上均存在正向外生偏移。保持该外生状态不变，若将温度从 \(750^\circ C\) 降至 \(700^\circ C\)，中间相指标 \(M\) 的反事实均值将从事实值 \(4.80\) 降至约 \(4.50\)，并进一步使目标产率 \(Y\) 从约 \(9.10\) 降至约 \(8.45\)。候选干预相对于该对照的反事实优势均值为 \(0.65\)，95% 可信区间为 \([0.06,1.24]\)，候选优于对照的后验概率约为 \(0.985\)。因此，在该事实情境下，\(750^\circ C\) 相比 \(700^\circ C\) 具有较稳定的反事实优势。

该解释的所有数值都来自 SCM-GP 计算结果，而不是 LLM 自行推断。

---

## 10. 实现条件与限制

严格 SCM-GP 版本虽然因果语义更强，但实现要求更高。

### 10.1 必要条件

1. 已知或可信的因果图 \(G\)；
2. 可观测关键中介变量或上下文变量；
3. 每个结构方程有足够样本进行 GP 建模；
4. 外生噪声模型可设定；
5. 能进行 GP posterior sampling；
6. 能沿因果图拓扑顺序进行前向传播；
7. 对未观测变量有合适的推断方法。

### 10.2 主要限制

1. 若中介变量不可观测，abduction 会不充分；
2. 若结构方程数据不足，GP-SCM posterior 会高度不确定；
3. 若外生噪声非加性，残差式 abduction 不再适用；
4. 若存在潜在混杂，需要显式建模共享外生变量；
5. 若图结构错误，反事实解释可能系统性偏差；
6. 高维系统中逐节点 GP 建模计算成本较高。

### 10.3 实用建议

论文中可采用双层策略：

- 主方法：posterior interventional contrastive reranking；
- 严格扩展：SCM-GP counterfactual contrast engine；
- 小型 synthetic SCM 实验完整展示 abduction-action-prediction；
- 大型 benchmark 使用更稳健的后验干预对照版本。

这样既能保证方法可落地，又能保留严格反事实语义。

---

## 11. 推荐写入论文的方法表述

### 中文表述

> 在严格 SCM-GP 版本中，本文对每个结构机制 \(f_j\) 分别放置高斯过程先验，并通过 abduction-action-prediction 执行反事实推理。给定事实情境 \(e\)，首先在结构函数后验样本条件下反推出外生噪声 \(U\) 的后验样本；随后分别将可干预变量的结构方程替换为候选干预 \(a\) 和对照干预 \(b\)，同时保持同一个外生噪声样本不变；最后沿因果图拓扑顺序前向传播，得到配对的 \(Y_a^{CF}\) 与 \(Y_b^{CF}\) 样本，并据此估计反事实差异 \(\Delta_{a,b}^{CF}(e)\) 的后验分布。

### 英文表述

> In the strict SCM-GP variant, each structural mechanism \(f_j\) is endowed with a Gaussian process prior, and counterfactual inference is performed through abduction, action, and prediction. Given a factual context \(e\), we first infer posterior samples of the exogenous variables \(U\) by conditioning on the observed factual variables and sampled structural functions. For each posterior sample, we then replace the structural equations of the intervened variables with candidate intervention \(a\) and contrast intervention \(b\), respectively, while keeping the same exogenous noise sample. Forward propagation along the causal graph yields paired samples \(Y_a^{CF,(s)}\) and \(Y_b^{CF,(s)}\), from which we estimate the posterior distribution of \(\Delta_{a,b}^{CF}(e)=Y_a^{CF}(e)-Y_b^{CF}(e)\).

---

## 12. 最终建议

严格 SCM-GP 版本是你们方法中最能支撑“真正反事实”说法的部分。它的核心优势是：

1. 保留同一个事实情境的外生噪声；
2. 显式执行 abduction-action-prediction；
3. 能解释“对这个具体事实样本，如果采取另一个干预会怎样”；
4. 反事实差异具有后验分布、可信区间和胜率；
5. 可作为 LLM 选择对照后的因果计算引擎。

但它不应无条件替代简化版。最稳妥策略是：

> 将 posterior interventional contrast 作为主算法的可落地版本，将 strict SCM-GP counterfactual contrast 作为因果语义更强的扩展模块，并在小型可控 SCM 实验中展示其完整计算过程。

这样既能保证论文的可实现性，又能回应审稿人关于反事实语义严格性的质疑。

