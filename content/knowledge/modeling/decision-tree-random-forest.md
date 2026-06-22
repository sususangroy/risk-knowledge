---
title: "决策树与随机森林：风控建模中的规则提炼与特征筛选"
category: modeling
summary: 深入介绍决策树和随机森林在信贷风控中的核心原理、典型用法与优缺点——决策树用于规则提炼和分箱辅助（树深一般 2-3 层），随机森林用于特征筛选和精度提升，二者构成逻辑回归评分卡前的重要工具链。
tags: [决策树, 随机森林, 集成学习, 风控建模, 特征重要性, 规则提炼, WOE分箱]
---

## 一、决策树：像信贷审批官一样做决策

决策树（Decision Tree）的逻辑与信贷审批员的思维方式高度相似——通过一系列"是/否"问题逐步筛选风险，每一步都聚焦于区分好坏客户最有效的那个变量。

```
是否批准贷款？
│
├── 近 12 个月逾期次数 > 2 次？
│   ├── 是 → 直接拒绝
│   └── 否 → 继续判断
│       ├── 负债收入比 > 60%？
│       │   ├── 是 → 拒绝
│       │   └── 否 → 月收入 > 5,000 元？
│       │       ├── 是 → 通过
│       │       └── 否 → 拒绝
```

这棵树深度为 3，逻辑清晰，可以直接翻译为业务规则，是决策树在风控中最核心的价值所在。

---

## 二、分裂准则：树如何"学习"切分规则

### 基尼系数（Gini Impurity）

CART 决策树（sklearn 默认）使用基尼系数衡量节点纯度：

```
Gini = 1 - Σ p_k²

  p_k：节点中第 k 类样本的占比
```

- 节点全为坏客户或全为好客户 → Gini = 0，最纯
- 节点好坏各半 → Gini = 0.5，最混
- 每次分裂选择**加权平均 Gini 最小**的切分方式

### 信息增益（Information Gain）

ID3/C4.5 算法使用信息熵作为分裂标准：

```
熵（Entropy） = -Σ p_k × log₂(p_k)

信息增益 = 分裂前的熵 - 分裂后的加权平均熵
```

选择信息增益最大的特征进行分裂。

**实践选择**：风控建模中通常使用 CART（基尼系数），sklearn 的 `DecisionTreeClassifier` 和 `RandomForestClassifier` 均默认使用 Gini。

---

## 三、风控中的核心参数与推荐设置

### 为什么风控决策树深度通常只有 2-3 层

这是风控领域与通用机器学习最显著的差异之一：

| 出发点 | 说明 |
|--------|------|
| **监管可解释性** | 银行、消费金融公司的风控模型需要向监管机构解释拒贷原因，超过 3 层的树规则已难以向审核人员口头描述 |
| **业务可实施性** | 提炼出的规则需要进入系统策略或人工审批流程，规则过于复杂会增加执行成本和出错概率 |
| **稳定性要求** | 层数越深，叶节点样本越少，规则越依赖训练数据的局部噪声，上线后稳定性差 |
| **防止过拟合** | 信贷数据好坏比通常为 10:1 到 50:1，深树容易对稀有的坏样本过拟合 |

**经验规则**：
- **规则提炼用途**：max_depth = 2 或 3，要求输出直接可用的业务策略
- **辅助分箱用途**：max_depth = 4 或 5，稍深以找到更细致的切分点
- **特征初筛用途**：max_depth = 6–8，允许稍深以充分捕捉特征关系

### 关键参数详解

```python
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    # ===== 核心复杂度控制 =====
    max_depth=3,              # 树的最大深度（风控规则提炼通常 2-3）
    min_samples_split=100,    # 分裂节点所需的最小样本数
    min_samples_leaf=50,      # 叶节点最小样本数（防止规则建立在极少样本上）

    # ===== 分裂准则 =====
    criterion='gini',         # 分裂标准（gini 或 entropy）
    max_features=None,        # None = 使用全部特征（单棵树一般不限）

    # ===== 样本不平衡 =====
    class_weight='balanced',  # 自动按好坏比调整样本权重

    random_state=42
)
```

**min_samples_leaf 的重要性**：

在信贷数据中，如果叶节点只有 20 个样本，其中 3 个坏客户，坏率 15%——但这 15% 的估计极不稳定，换一批数据可能变成 5% 或 30%。设置 `min_samples_leaf=50`（乃至更大）能保证每条规则有足够的统计支撑。

---

## 四、决策树在风控中的三大典型用法

### 用法一：直接提炼业务规则

决策树的分裂规则可直接转译为系统策略，无需复杂解释：

```python
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

# 文本格式输出规则
rules = export_text(dt_model, feature_names=feature_names)
print(rules)
# 输出示例：
# |--- 近12月逾期次数 <= 1.50
# |   |--- 负债收入比 <= 0.60
# |   |   |--- class: 好客户（通过）
# |   |   |--- class: 坏客户（拒绝）
# |   |--- 负债收入比 > 0.60
# |   |   |--- class: 坏客户（拒绝）
# |--- 近12月逾期次数 > 1.50
# |   |--- class: 坏客户（拒绝）

# 可视化
plt.figure(figsize=(12, 6))
plot_tree(dt_model, feature_names=feature_names,
          class_names=['好客户', '坏客户'],
          filled=True, rounded=True)
plt.show()
```

### 用法二：辅助 WOE 分箱

人工分箱容易主观，决策树可以自动找到统计意义上最优的分割点，再结合业务经验微调：

```python
# 用单变量决策树找最优切分点
def find_optimal_splits(df, feature, target, max_depth=4):
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=50,
        class_weight='balanced'
    )
    dt.fit(df[[feature]], df[target])

    # 提取分裂阈值
    tree = dt.tree_
    thresholds = sorted(set(
        tree.threshold[tree.threshold != -2]  # -2 是叶节点标记
    ))
    print(f"{feature} 的建议分割点: {[round(t, 2) for t in thresholds]}")
    return thresholds
```

### 用法三：特征重要性初筛

在进入 WOE 编码和逻辑回归建模前，先用决策树快速排除明显无信息量的特征：

```python
import pandas as pd

importance_df = pd.DataFrame({
    'feature':    feature_names,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

# 去掉重要性极低的特征（通常 < 0.01 可先排除）
useful_features = importance_df[importance_df['importance'] > 0.01]['feature'].tolist()
```

---

## 五、决策树的优缺点

### 优点

| 优点 | 说明 |
|------|------|
| **规则直接可读** | 树结构可直接翻译为"如果…则…"规则，无需额外解释 |
| **无需特征工程** | 不需要归一化、标准化，对异常值和缺失值鲁棒性较好 |
| **捕捉非线性关系** | 天然处理变量与违约率之间的非线性、非单调关系 |
| **无需假设分布** | 不要求变量服从正态分布，适合信贷数据的实际分布 |
| **计算速度快** | 训练和预测都很快，适合快速迭代和探索 |

### 缺点

| 缺点 | 说明 |
|------|------|
| **高方差、不稳定** | 训练数据稍有变化，树的结构可能大幅改变 |
| **单棵树精度有限** | 单棵浅树（深度 2-3）的区分度通常不如集成模型 |
| **容易过拟合** | 深树会对训练数据噪声过拟合，特别是坏样本稀少时 |
| **类别不平衡敏感** | 好坏样本极度不平衡时，不设 class_weight 会导致规则偏向好客户 |
| **无法输出概率** | 叶节点的违约率估计在样本少时极不稳定，不适合直接作为评分输出 |

---

## 六、随机森林：多棵树的集体智慧

单棵决策树的高方差问题由随机森林（Random Forest）通过两个核心机制解决：

### 两重随机性

**1. Bootstrap 采样（Bagging）**

每棵树从原始训练集中**有放回地随机抽取约 63%** 的样本（约 37% 的样本不被选中，即 OOB 样本）。不同的树见到不同的数据子集，相互之间的相关性降低，集成后方差显著减小。

**2. 随机特征选择**

每个节点分裂时，不从全部特征中选最优切分，而是从**随机抽取的 √n 个特征**（n = 总特征数）中选择最优切分。这使得即使在强特征存在的情况下，弱特征也有机会参与构建，各棵树的差异性更大，集成效果更好。

### 预测机制

```
分类任务：多数投票
  → 超过半数的树预测为"坏客户"，则最终判为坏客户

概率输出：各树预测概率取平均
  → 更稳定的违约概率估计
```

---

## 七、随机森林的核心参数

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    # ===== 集成规模 =====
    n_estimators=300,         # 树的数量（越多越稳定，但边际收益递减）

    # ===== 单棵树复杂度 =====
    max_depth=8,              # 随机森林用于精度，可以比单棵决策树深
    min_samples_leaf=30,      # 叶节点最小样本数
    max_features='sqrt',      # 每次分裂随机选取的特征数（分类任务推荐）

    # ===== 样本不平衡 =====
    class_weight='balanced',  # 自动调整好坏样本权重
    # 或：class_weight='balanced_subsample'（每棵树单独计算权重）

    # ===== 评估与工程 =====
    oob_score=True,           # 开启 OOB 评估（相当于免费的验证集）
    n_jobs=-1,                # 并行训练
    random_state=42
)

rf_model.fit(X_train, y_train)
print(f"OOB Score: {rf_model.oob_score_:.4f}")
```

**参数调优优先级**：

| 参数 | 调优建议 |
|------|---------|
| n_estimators | 先设 100，若 OOB Score 仍在提升则加至 300–500 |
| max_depth | 通常 6–10；太深容易过拟合，太浅精度不足 |
| min_samples_leaf | 信贷数据建议 30–50，样本量大时可降至 20 |
| max_features | 分类任务用 'sqrt'，特征极多时可尝试 0.5 |

---

## 八、OOB 评估：不需要专门划出验证集

Bootstrap 采样使得每棵树约有 37% 的样本未被训练（OOB 样本），可直接用于评估：

```python
# OOB 预测概率（对每个样本，用没见过它的那些树来预测）
oob_proba = rf_model.oob_decision_function_[:, 1]

from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

oob_auc = roc_auc_score(y_train, oob_proba)
ks_stat, _ = ks_2samp(oob_proba[y_train == 1], oob_proba[y_train == 0])

print(f"OOB AUC: {oob_auc:.4f}")
print(f"OOB KS:  {ks_stat:.4f}")
```

OOB 评估与留出验证集的结果高度相关，在样本量有限时尤其有价值。

---

## 九、随机森林用于特征筛选

在风控建模中，随机森林最常见的用途是**在进入 WOE 编码和逻辑回归建模前做特征预筛选**：

```python
# Step 1：用随机森林跑出特征重要性
rf_selector = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=30,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf_selector.fit(X_train, y_train)

importance_series = pd.Series(
    rf_selector.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

# Step 2：保留重要性前 N 名（或重要性 > 阈值）
top_features = importance_series.head(50).index.tolist()

# Step 3：对筛选后的特征做 WOE 编码 → 逻辑回归
```

**注意**：随机森林特征重要性基于 Gini 下降量，存在对高基数（取值种类多）特征的天然偏倚，数值型连续特征的重要性通常会高于等效的类别型特征。建议结合 IV 值共同判断。

---

## 十、随机森林的优缺点

### 优点

| 优点 | 说明 |
|------|------|
| **精度高、稳定** | 集成大量树后方差显著降低，对数据扰动不敏感 |
| **天然抗过拟合** | 随机采样 + 随机特征使单棵树互相独立，不容易同时过拟合 |
| **特征重要性可用** | 批量筛选数百个候选特征时效率极高 |
| **处理高维特征** | 特征数量达数百乃至数千时仍能有效工作 |
| **OOB 免费评估** | 无需额外划分验证集即可得到可靠的模型性能估计 |
| **类别不平衡友好** | class_weight='balanced' 配合 bootstrap，处理好坏不平衡效果好 |

### 缺点

| 缺点 | 说明 |
|------|------|
| **不可解释** | 数百棵树的集成无法直接提炼规则，不满足监管对逐变量解释的要求 |
| **训练较慢** | 相比单棵决策树，训练时间显著增加（可并行缓解） |
| **内存占用大** | n_estimators=500 的模型存储可能达数 GB |
| **概率校准差** | 输出的概率分布偏保守（集中在中间区域），如需精确概率需额外校准 |
| **特征重要性有偏** | 对高基数特征偏高估，需结合 IV 值交叉验证 |

---

## 十一、决策树 vs 随机森林：如何选择

| 维度 | 决策树（浅，2-3 层） | 随机森林 |
|------|---------------------|---------|
| **主要用途** | 规则提炼、分箱辅助 | 特征筛选、精度建模 |
| **区分度（KS）** | 低（通常 0.2–0.35） | 中高（通常 0.35–0.5） |
| **可解释性** | 最高（规则直接可读） | 低（需 SHAP 等工具辅助） |
| **稳定性** | 低（高方差） | 高（集成降低方差） |
| **监管友好度** | 高（可直接向监管解释） | 低（黑盒，需配合解释工具） |
| **训练速度** | 极快 | 较慢 |
| **适用数据量** | 中小（万级别已够） | 中大（十万级别更稳） |

### 风控标准工作流

```
原始候选特征（数百个）
    │
    ├─ 随机森林 → 特征重要性排序 → 保留 Top 50–100
    │
    ├─ IV 计算 → 剔除 IV < 0.02 的变量（与上步交叉验证）
    │
    ├─ 决策树（深度 4-5）→ 辅助确认 WOE 最优分割点
    │
    ├─ WOE 编码 + VIF 共线性检验
    │
    └─ 逻辑回归评分卡
```

---

## 十二、实战常见坑

1. **深度设太大做规则提炼**：`max_depth=10` 的决策树输出的规则复杂度堪比 XGBoost，失去了可解释优势，深度 2-3 才是风控的正确姿势

2. **忘记设 class_weight**：好坏 20:1 不设权重，模型会把所有人预测为好客户，KS 接近 0

3. **用随机森林特征重要性直接排除变量**：重要性低不等于 IV 低，建议同时计算 IV，两个维度都低的特征再剔除

4. **随机森林当主模型上线**：随机森林无法提供逐变量的系数解释，在强监管场景（银行、持牌消金）不宜直接作为评分模型，应作为特征筛选工具

5. **OOB Score 替代测试集**：OOB 评估存在时序问题，当训练数据有时间顺序时（信贷场景通常如此），OOB 样本混在了不同时间段，无法替代按时间划分的真实测试集
