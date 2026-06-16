# 语言引导的可行反事实对照选择决策层用于可解释因果贝叶斯优化：方法可行性与研究方案报告

**报告类型**：学术方法审阅、理论澄清与研究方案重构  
**语言**：中文  
**主题关键词**：因果贝叶斯优化、反事实对照、大语言模型、结构因果模型、高斯过程、可解释序列干预优化  
**日期**：2026-06-15  

---

## 摘要

本报告对“语言引导的反事实对照决策层用于可解释因果贝叶斯优化”这一方法进行系统审阅与重构。总体判断是：该研究方向具有较强可行性和学术辨识度，但需要对方法主张进行严格收缩，避免把“后验干预对照”误称为“严格个体化反事实”，也避免让大语言模型承担因果推理、数值预测或优化决策职责。

建议将方法从原始表述中的 **LLM-generated counterfactual intervention** 修正为：

> **LLM-guided feasible contrast selection for explainable causal Bayesian optimization**  
> 即：语言引导的可行对照选择决策层用于可解释因果贝叶斯优化。

核心思想是：标准 CBO acquisition 先产生高价值候选池；算法根据实验空间、历史样本、后验摘要和因果图构造合法数值对照池；LLM 仅从该合法池中选择具有机制比较价值、解释价值和领域意义的对照；验证器过滤非法或无意义对照；SCM-GP 或 GP posterior 计算对照差异；最终只在 near-tie candidate pool 内进行保守重排序，并由 LLM 根据结构化计算结果生成忠实解释。

因此，推荐的角色分工是：

> **LLM selects, SCM answers, CBO decides, LLM explains.**

---

## 1. 研究背景与核心问题

贝叶斯优化（Bayesian Optimization, BO）是一类面向昂贵黑箱函数的序列优化方法。其典型做法是使用高斯过程等 surrogate model 建模目标函数，并通过 acquisition function 在探索与利用之间权衡。

因果贝叶斯优化（Causal Bayesian Optimization, CBO）进一步将优化对象放入结构因果模型中，利用因果图、do-calculus、观测数据与干预数据来选择最优干预。经典 CBO 工作强调 CBO 将 BO 从“相互独立的黑箱输入变量”推广到“具有因果依赖结构的干预优化问题”。后续工作包括 Dynamic CBO、Model-based CBO、Functional CBO、Constrained CBO 以及外生变量分布学习等方向。

然而，现有 CBO 方法主要回答：

- 如何利用因果图降低干预搜索空间；
- 如何结合观测数据与干预数据；
- 如何估计干预效应；
- 如何设计或优化 acquisition function；
- 如何在有限预算内提升最优干预搜索效率。

但在许多真实实验中，大多数候选干预都是技术上可执行的。真正困难的问题是：

> 当多个可行干预具有相近 acquisition value 时，哪个干预更值得优先实验？

标准 EI、CEI、UCB 或 CBO acquisition 通常不能直接回答以下机制性问题：

- 为什么选择候选 \(a\)，而不是候选 \(b\)？
- 当前候选是否能区分两个竞争性机制假设？
- 当前收益是否来自稳定的因果路径？
- 当前候选相对于历史最优点是否真的有额外价值？
- 当前实验是否能回答领域专家真正关心的问题？

这正是本方法试图补充的空间。

---

## 2. 关键概念澄清：后验干预对照不等于严格反事实

### 2.1 后验干预对照

若 CBO surrogate 学习的是：

\[
g(x)=\mathbb E[Y\mid do(X=x)]
\]

那么比较两个干预 \(a,b\) 得到：

\[
\Delta_{a,b}^{IC}=g(a)-g(b)
\]

该对象的含义是：

> 在总体或后验意义下，执行干预 \(a\) 相比执行干预 \(b\) 的平均目标效果差异。

若 \(g\) 由 GP posterior 建模，则：

\[
\Delta_{a,b}^{IC}\mid D_t
\sim
\mathcal N(
\mu_t(a)-\mu_t(b),
\sigma_t^2(a)+\sigma_t^2(b)-2k_t(a,b)
)
\]

这可以称为 **posterior interventional contrast**，即后验干预对照。

### 2.2 严格 SCM 反事实

严格的 Pearl 式反事实问的是：

\[
Y_{do(X=b)}(e)
\]

即：

> 对同一个已经发生的事实情境 \(e\)，如果当时执行的不是 \(a\)，而是 \(b\)，结果会怎样？

这需要执行三步：

1. **Abduction**：根据事实情境反推出外生噪声：
   \[
   p(U\mid e,D_t,G)
   \]

2. **Action**：将被干预变量的结构方程替换为常数：
   \[
   X:=b
   \]

3. **Prediction**：保持同一外生噪声，沿因果图前向传播：
   \[
   p(Y_b^{CF}\mid e,D_t,G)
   \]

因此，没有 abduction-action-prediction，就不能声称完成了严格个体化反事实推理。

### 2.3 对论文写法的建议

如果主算法只计算 \(g(a)-g(b)\)，则论文应称之为：

> posterior interventional contrastive reranking

而不是 full counterfactual inference。

如果要声称严格反事实，则必须建立 GP-SCM，对每个结构方程建模：

\[
X_j=f_j(PA_j,U_j)
\]

并推断：

\[
p(U\mid e,D_t,G)
\]

---

## 3. 数值例子：为什么两者不同

设一个药物剂量系统：

\[
M=X+U_M
\]

\[
Y=M+HX+U_Y
\]

其中 \(X\) 是药物剂量，\(M\) 是中介生物标志物，\(Y\) 是治疗效果，\(H\) 是患者敏感性。

若总体中：

\[
\mathbb E[H]=1,\quad \mathbb E[U_M]=0,\quad \mathbb E[U_Y]=0
\]

则：

\[
\mathbb E[Y\mid do(X=1)]=2
\]

\[
\mathbb E[Y\mid do(X=0)]=0
\]

所以后验或总体干预对照为：

\[
g(1)-g(0)=2
\]

现在考虑一个具体事实情境：实际执行 \(X=1\)，观察到：

\[
M=1.2,\quad Y=6.2
\]

由 \(M=X+U_M\) 得：

\[
U_M=0.2
\]

若设 \(U_Y=0\)，则：

\[
6.2=1.2+H
\]

所以：

\[
H=5
\]

该样本是高敏感性个体。若问该同一情境下执行 \(X=0\) 的反事实结果，则保持：

\[
H=5,\quad U_M=0.2,\quad U_Y=0
\]

执行 \(do(X=0)\)：

\[
M_{X=0}=0.2
\]

\[
Y_{X=0}=0.2
\]

所以个体化反事实差异为：

\[
6.2-0.2=6.0
\]

这与总体干预对照 \(2.0\) 完全不同。该例说明：**比较两个 do-world 的平均效果，不等于回答同一个事实情境下的反事实结果。**

---

## 4. 方法重构：LLM-guided feasible contrast selection

### 4.1 为什么不能让 LLM 自由生成候选

真实实验变量通常是固定数值、离散水平或连续区间，例如：

| 变量 | 含义 | 允许取值 |
|---|---|---|
| \(X_1\) | 温度 | \(\{600,650,700,750,800\}\) ℃ |
| \(X_2\) | 催化剂比例 | \(\{0.05,0.10,0.15,0.20\}\) |
| \(X_3\) | 反应时间 | \(\{1,2,3,4\}\) h |

因此，算法层不应出现：

```json
{"temperature": "high", "catalyst_ratio": "medium"}
```

而应使用数值化干预：

```json
{"temperature": 750, "catalyst_ratio": 0.15, "reaction_time": 3}
```

high/low 只能作为解释层语言，不能作为正式算法定义。

### 4.2 合法对照池

对每个 CBO 高价值候选 \(a\)，算法先构造合法数值对照池：

\[
\mathcal P_t(a)
=
\mathcal P_{local}(a)
\cup
\mathcal P_{history}(a)
\cup
\mathcal P_{posterior}(a)
\cup
\mathcal P_{mechanism}(a)
\cup
\mathcal P_{cost}(a)
\]

其中：

- \(\mathcal P_{local}(a)\)：单变量或少变量局部扰动；
- \(\mathcal P_{history}(a)\)：历史最优点、近邻高表现点、失败点；
- \(\mathcal P_{posterior}(a)\)：后验均值接近或方差较高的点；
- \(\mathcal P_{mechanism}(a)\)：作用于相同中介、竞争路径或协同机制的点；
- \(\mathcal P_{cost}(a)\)：低成本、低剂量、低风险但预测效果接近的点。

然后 LLM 只从中选择：

\[
B_{LLM}(a)\subseteq \mathcal P_t(a)
\]

### 4.3 LLM 的形式化角色

推荐定义：

\[
B_{LLM}(a)
=
Select_{LLM}(a,\mathcal P_t(a),G,D_t,S_t,\mathcal T)
\]

其中：

- \(a\)：当前候选；
- \(\mathcal P_t(a)\)：合法数值对照池；
- \(G\)：因果图；
- \(D_t\)：历史实验数据；
- \(S_t\)：GP/CBO 后验摘要；
- \(\mathcal T\)：对照类型集合；
- \(B_{LLM}(a)\)：LLM 选择出的对照集合。

LLM 的输出应是结构化 JSON，例如：

```json
{
  "candidate_id": "A",
  "selected_contrasts": [
    {
      "contrast_id": "B1",
      "contrast_type": "single_variable_temperature_reduction",
      "changed_variables": ["temperature"],
      "rationale": "This contrast isolates whether increasing temperature from 700 to 750 is necessary while keeping catalyst ratio and reaction time fixed."
    },
    {
      "contrast_id": "B2",
      "contrast_type": "catalyst_dose_reduction",
      "changed_variables": ["catalyst_ratio"],
      "rationale": "This contrast tests whether the higher catalyst ratio is necessary under the same temperature and reaction time."
    }
  ]
}
```

---

## 5. 对照类型设计

建议将 LLM 的选择限制在有限 taxonomy：

| 对照类型 | 作用 |
|---|---|
| single-variable perturbation | 隔离单个变量的边际贡献 |
| dose/cost reduction | 判断低剂量、低成本或低风险是否足够 |
| historical incumbent comparison | 解释为什么不采用历史最优点 |
| nearby high-performing comparison | 判断新候选是否优于近邻好点 |
| negative control comparison | 解释某关键变量改变是否导致失败 |
| mechanism-path contrast | 区分不同因果路径贡献 |
| synergy/saturation test | 判断变量组合是否存在协同或饱和 |
| uncertainty-driven contrast | 回答当前 posterior 中不确定但机制相关的问题 |
| constraint-risk contrast | 比较更安全或更易执行的替代方案 |

这种 taxonomy 的作用是提高可复现性，避免 LLM 自由发挥。

---

## 6. 因果查询验证器

LLM 选择出的对照必须经过验证器：

\[
V(q,G,\Omega,D_t,S_t)\in\{0,1\}
\]

验证器检查：

1. 对照只改变可干预变量；
2. 所有变量取值在实验空间 \(\Omega\) 内；
3. 不违反领域、安全、成本或工艺约束；
4. 与候选 \(a\) 不相同；
5. 满足最小差异性，除非对照类型是协同测试；
6. 未远离 surrogate 支持区域；
7. 不把中介变量或结果变量当作干预变量；
8. LLM rationale 与实际变化变量一致；
9. 去除重复对照；
10. 控制对照类型多样性。

只有通过验证的对照才能进入 GP/SCM posterior 计算。

---

## 7. 反事实/对照评分

### 7.1 后验可区分性

对候选 \(a\) 和对照 \(b\)，定义：

\[
Disc_t(a,b)
=
rac{|\mu_t(a)-\mu_t(b)|}
{\sqrt{\sigma_t^2(a)+\sigma_t^2(b)-2k_t(a,b)+\eta}}
\]

其中 \(\eta>0\) 是数值稳定项。

候选级别：

\[
Disc_t(a)=\max_{b\in B_{LLM}(a)}Disc_t(a,b)
\]

### 7.2 对照不确定性

定义：

\[
Unc_t(a,b)=
\sigma_t^2(a)+\sigma_t^2(b)-2k_t(a,b)
\]

该项表示候选与对照之间的后验差异仍有多大不确定性。它不应被简单解释为“方差越大越好”，而应与机制相关性和 near-tie 候选限制共同使用。

### 7.3 综合分数

第一版建议使用：

\[
CFScore_t(a)
=
\lambda_1 z(Disc_t(a))
+
\lambda_2 z(Unc_t(a))
\]

其中 \(z(\cdot)\) 表示标准化。

不建议第一版将“机制稳定性”作为主评分项，除非能够严格定义 path-specific contribution 或 posterior sample 下的路径归因。

---

## 8. Near-tie 保守重排序

定义基础 acquisition 的最大值：

\[
lpha_t^\star=\max_{x\in\Omega}lpha_t^{CBO}(x)
\]

near-tie 候选池为：

\[
\mathcal C_t(\delta_t)
=
\{x\in\Omega:lpha_t^{CBO}(x)\ge lpha_t^\star-\delta_t\}
\]

最终选择：

\[
x_{t+1}
\in
rg\max_{a\in\mathcal C_t(\delta_t)}
CFScore_t(a)
\]

该设计保证：

\[
0
\le
lpha_t^\star-lpha_t^{CBO}(x_{t+1})
\le
\delta_t
\]

若 \(\delta_t	o0\)，则方法在 acquisition value 意义下渐近接近基础 CBO 的 acquisition-optimal 行为。

因此，论文不应声称提出具有完整 regret bound 的新 acquisition，而应声称：

> 该方法是一个保守 decision layer，在 near-tie candidate pool 内引入机制对照价值和解释价值，同时保持基础 CBO 主循环不被破坏。

---

## 9. 严格 GP-SCM 反事实扩展

若要支持严格个体化反事实，应对每个结构方程建模：

\[
X_j=f_j(PA_j,U_j)
\]

可设：

\[
f_j\sim GP(m_j,k_j)
\]

然后执行：

### 9.1 Abduction

\[
p(U\mid e,D_t,G)
\]

在 additive noise model 下，可用残差近似：

\[
\hat U_j=x_j-\hat f_j(pa_j)
\]

更严格版本应对结构函数和外生噪声联合采样。

### 9.2 Action

替换干预变量结构方程：

\[
X_S:=a
\]

或：

\[
X_S:=b
\]

### 9.3 Prediction

沿因果图拓扑顺序传播：

\[
X_j^{CF}=f_j(PA_j^{CF},U_j)
\]

最终得到：

\[
p(Y_a^{CF}-Y_b^{CF}\mid e,D_t,G)
\]

建议主文中写：

> Our main implementation uses posterior interventional contrasts for robustness and compatibility with standard CBO. When structural mechanisms and contextual observations are available, the same contrastive decision layer can be instantiated with a GP-SCM counterfactual engine that performs abduction-action-prediction.

---

## 10. 算法流程

**Algorithm: LLM-Guided Feasible Contrast Selection for CBO**

输入：

- 因果图 \(G\)；
- 可干预变量集合 \(\mathcal I\)；
- 可行空间 \(\Omega\)；
- 历史数据 \(D_t\)；
- 基础 CBO acquisition \(lpha_t^{CBO}\)；
- GP/SCM posterior summary \(S_t\)；
- LLM contrast selector；
- 验证器 \(V\)；
- 实验预算 \(T\)。

步骤：

1. 初始化 \(D_0\)。
2. 对 \(t=0,\dots,T-1\)：
   1. 使用 \(D_t\) 更新 CBO surrogate。
   2. 计算 \(lpha_t^{CBO}(x)\)。
   3. 构造 near-tie 候选池 \(\mathcal C_t(\delta_t)\)。
   4. 对每个 \(a\in\mathcal C_t(\delta_t)\)：
      1. 构造合法数值对照池 \(\mathcal P_t(a)\)。
      2. LLM 从 \(\mathcal P_t(a)\) 中选择语义对照 \(B_{LLM}(a)\)。
      3. 验证器过滤非法对照。
      4. GP/SCM posterior 计算 \(\Delta_{a,b}\)。
      5. 计算 \(CFScore_t(a)\)。
   5. 在 \(\mathcal C_t(\delta_t)\) 内选择 \(CFScore_t(a)\) 最高的候选。
   6. 执行干预并观测 \(y_{t+1}\)。
   7. 更新 \(D_{t+1}=D_t\cup\{(x_{t+1},y_{t+1})\}\)。
   8. LLM 基于结构化计算结果生成解释。
3. 返回当前最优干预估计。

---

## 11. 实验设计

### 11.1 实验目标

实验应验证：

1. 方法是否提升或至少不显著损害 CBO 优化性能；
2. LLM 选择的对照是否比随机、最近邻或规则对照更有机制价值；
3. SCM-GP/GP posterior 解释是否比普通 CBO 解释更忠实；
4. near-tie reranking 是否比全局 CFScore 替代 acquisition 更安全。

### 11.2 Benchmark

建议设置：

1. **标准 synthetic SCM**：chain、fork、mediator、collider、高维稀疏图；
2. **机制歧义 benchmark**：多个 acquisition 接近候选对应不同机制路径；
3. **协同效应 benchmark**：变量组合存在交互、饱和或负协同；
4. **剂量/成本权衡 benchmark**：高收益候选和低成本候选 acquisition 接近；
5. **图结构扰动 benchmark**：测试轻微图错误下鲁棒性。

### 11.3 Baseline

| Baseline | 目的 |
|---|---|
| Random Search | 最低基线 |
| Standard BO | 检验忽略因果图的影响 |
| Standard CBO | 主性能基线 |
| CBO + Random Contrast Reranking | 检验是否只是任意重排序 |
| CBO + Nearest Contrast Reranking | 检验最近邻对照是否足够 |
| CBO + One-variable Rule Contrast | 检验简单规则是否可替代 LLM |
| CBO + LLM Direct Scoring | 证明 LLM 直接打分不可靠 |
| Proposed w/o Validator | 验证验证器必要性 |
| Proposed w/o Near-tie | 验证保守候选池必要性 |
| Proposed Method | 完整方法 |

### 11.4 指标

优化性能：

\[
r_T=f(x^\star)-f(\hat x_T)
\]

\[
R_T=\sum_{t=1}^{T}[f(x^\star)-f(x_t)]
\]

以及 best-found outcome、cost-normalized regret。

对照质量：

- valid contrast rate；
- contrast relevance；
- mechanism coverage；
- contrast diversity；
- minimality；
- support-region compliance。

解释质量：

- explanation fidelity；
- specificity；
- uncertainty awareness；
- hallucination rate；
- expert usefulness score；
- auditability。

安全性：

- performance degradation relative to CBO；
- sensitivity to \(\delta_t\)；
- robustness under graph misspecification；
- sensitivity to LLM model choice。

### 11.5 消融实验

至少包括：

1. 去掉 LLM，用随机对照；
2. 去掉 LLM，用最近邻对照；
3. 去掉 LLM，用单变量规则扰动；
4. 去掉 validator；
5. 去掉 near-tie 限制；
6. 固定 \(\delta\) vs 退火 \(\delta_t	o0\)；
7. 只使用 \(Disc_t\)；
8. 使用 \(Disc_t+Unc_t\)；
9. 不同候选池大小；
10. 不同 LLM；
11. 不同因果图质量；
12. posterior interventional contrast vs GP-SCM counterfactual extension。

---

## 12. 论文贡献建议

建议将贡献写成四点：

### C1. 语言引导的可行数值对照选择

提出一种面向真实实验变量的可行对照选择框架。不同于让 LLM 自由生成反事实，该方法先构造合法数值候选池，再让 LLM 选择具有语义和机制价值的对照。

### C2. 后验干预对照驱动的保守重排序

提出在 CBO near-tie candidate pool 内进行的对照重排序方法，不替代基础 acquisition，而是在高价值候选之间引入机制解释价值。

### C3. 严格区分干预对照与 SCM 反事实

明确区分 posterior interventional contrast 和 individualized SCM counterfactual，并提供 GP-SCM 扩展路径。

### C4. 忠实解释生成机制

LLM 只基于结构化计算结果生成解释，禁止输出未由 posterior 或 SCM 计算支持的因果断言。

---

## 13. 投稿风险与应对

### 风险 1：审稿人认为这不是真正反事实

应对：主动区分 interventional contrast 与 full SCM counterfactual。主算法称为 posterior interventional contrast，扩展版本称为 GP-SCM counterfactual engine。

### 风险 2：审稿人认为只是 reranking

应对：承认是 conservative decision layer，但强调候选来自 CBO、对照来自合法数值池、LLM 仅做语义选择、评分来自 posterior，且 acquisition suboptimality 被 \(\delta_t\) 控制。

### 风险 3：LLM 不可靠

应对：LLM 不做数值预测，不做因果推理，不直接决策；输出经 validator 过滤；加入 LLM direct scoring baseline。

### 风险 4：优化性能下降

应对：只在 near-tie pool 内重排序；使用 \(\delta_t	o0\)；报告 regret 和 performance degradation。

### 风险 5：解释幻觉

应对：LLM 只能读取结构化结果，强制报告 posterior mean、variance、credible interval 和 contrast type；引入 hallucination rate 与 explanation fidelity。

---

## 14. 推荐标题

中文标题：

1. 语言引导的可行对照选择决策层用于可解释因果贝叶斯优化
2. 大模型提问、因果模型回答：面向因果贝叶斯优化的可行反事实对照选择
3. 面向可解释因果贝叶斯优化的语言引导后验干预对照重排序
4. 基于语言模型语义对照选择的机制可解释因果贝叶斯优化

英文标题：

1. LLM-Guided Feasible Contrast Selection for Explainable Causal Bayesian Optimization
2. Language Models Select, Causal Models Answer: Contrastive Decision Layers for Causal Bayesian Optimization
3. Posterior Interventional Contrastive Reranking for Explainable Causal Bayesian Optimization
4. Language-Guided Contrastive Decision Making in Causal Bayesian Optimization

---

## 15. 推荐论文结构

1. Introduction  
   CBO 背景、现有不足、LLM 不应直接做因果推理、本文核心思想。

2. Related Work  
   BO、CBO、Model-based CBO、Functional CBO、LLM for BO、SCM counterfactual、GP-SCM。

3. Problem Formulation  
   SCM、CBO、posterior interventional contrast、individualized SCM counterfactual、near-tie reranking。

4. Method  
   base CBO candidate generation、feasible numeric contrast pool、LLM-guided contrast selection、validator、GP/SCM posterior evaluation、near-tie reranking、faithful explanation。

5. Theoretical Discussion  
   conservative decision layer、acquisition suboptimality bound、asymptotic degeneration、limitations。

6. Experiments  
   benchmark、baseline、optimization performance、contrast quality、explanation fidelity、ablation。

7. Discussion  
   适用场景、失败场景、图错误鲁棒性、GP-SCM 扩展。

8. Conclusion。

---

## 16. 可直接写入论文的关键表述

### 16.1 LLM 角色

> The LLM is not used as a causal reasoner, surrogate model, or optimizer. It acts as a constrained semantic contrast selector that chooses meaningful contrastive interventions from a validated feasible intervention pool. All causal validity checks and numerical effect evaluations are performed by the validator and the SCM-GP posterior.

### 16.2 反事实语义

> We distinguish posterior interventional contrasts from individualized SCM counterfactuals. The former compares \(g(a)-g(b)\), where \(g(x)=\mathbb E[Y\mid do(X=x)]\), and therefore marginalizes over exogenous variation. The latter requires a structural model, a factual context, and abduction over exogenous variables followed by action and prediction.

### 16.3 理论边界

> The proposed method does not replace the base CBO acquisition. It only reranks candidates within a near-tie set induced by the base acquisition. Therefore, if the selected candidate lies in \(\mathcal C_t(\delta_t)\), its acquisition suboptimality is bounded by \(\delta_t\).

### 16.4 数值变量

> In real experimental settings, intervention variables are numerical or constrained discrete values rather than linguistic levels such as high or low. Therefore, the contrast pool is constructed as a set of feasible numerical interventions, and the LLM only selects among candidate IDs with structured rationales.

---

## 17. 最终结论

该研究方向值得继续推进。它的核心价值不在于让 LLM 替代 CBO，也不在于构造一个难以证明的新 acquisition，而在于为 CBO 增加一个保守、可审计、机制对照驱动的解释与辅助决策层。

最稳妥的论文定位是：

> 在标准 CBO acquisition 产生的高价值候选池中，利用算法构造合法数值对照池，利用 LLM 选择有领域语义和机制比较价值的对照，再由 SCM-GP 或 GP posterior 计算后验对照差异，最终进行 near-tie reranking 和忠实解释生成。

按此定位，方法具备以下优点：

1. 保留基础 CBO 主循环；
2. 避免 LLM 直接因果推理导致的幻觉；
3. 适配真实实验中的数值变量；
4. 能解释“为什么选择 \(a\) 而不是 \(b\)”；
5. 可通过 contrast quality、explanation fidelity 和 regret 等指标系统验证；
6. 可扩展到严格 GP-SCM 反事实推理。

一句话总结：

> **LLM 的价值不是回答反事实，而是帮助提出值得回答的、合法的、数值化的机制对照问题。答案必须由 SCM-GP 或 GP posterior 给出。**

---

## 参考文献

1. Aglietti, V., Lu, X., Paleyes, A., & González, J. *Causal Bayesian Optimization*. AISTATS 2020.  
   https://proceedings.mlr.press/v108/aglietti20a.html

2. Aglietti, V., Dhir, N., González, J., & Damoulas, T. *Dynamic Causal Bayesian Optimization*. NeurIPS 2021.  
   https://arxiv.org/abs/2110.13891

3. Sussex, S., Makarova, A., & Krause, A. *Model-based Causal Bayesian Optimization*. ICLR 2023.  
   https://arxiv.org/abs/2211.10257

4. Gultchin, L., Aglietti, V., Bellot, A., & Chiappa, S. *Functional Causal Bayesian Optimization*. UAI 2023.  
   https://arxiv.org/abs/2306.06409

5. Ren, S., & Qian, X. *Causal Bayesian Optimization via Exogenous Distribution Learning*. 2024.  
   https://arxiv.org/abs/2402.02277

6. Liu, T., et al. *Large Language Models to Enhance Bayesian Optimization*. ICLR 2024.  
   https://openreview.net/forum?id=OOxotBmGol

7. Pawlowski, N., Castro, D. C., & Glocker, B. *Deep Structural Causal Models for Tractable Counterfactual Inference*. NeurIPS 2020.  
   https://arxiv.org/abs/2006.06485

8. Giudice, E., Kuipers, J., & Moffa, G. *Bayesian Causal Inference with Gaussian Process Networks*. 2024.  
   https://arxiv.org/abs/2402.00623

9. Witty, S., Takatsu, K., Jensen, D., & Mansinghka, V. *Causal Inference using Gaussian Processes with Structured Latent Confounders*. ICML 2020.  
   https://proceedings.mlr.press/v119/witty20a.html

10. Pearl, J. *Causality: Models, Reasoning, and Inference*. Cambridge University Press, 2009.
