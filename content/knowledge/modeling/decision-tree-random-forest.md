---
order: 8
title: "决策树与随机森林：风控建模中的规则提炼与特征筛选"
category: modeling
summary: 深入介绍决策树和随机森林在信贷风控中的核心原理、典型用法与优缺点——决策树用于规则提炼和分箱辅助（树深一般 2-3 层），随机森林用于特征筛选和精度提升，二者构成逻辑回归评分卡前的重要工具链。
tags: [决策树, 随机森林, 集成学习, 风控建模, 特征重要性, 规则提炼, WOE分箱]
---

## 一、决策树

### 核心原理

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

### 分裂准则

树模型在训练时，从所有特征和所有可能的切分点中，自动找到"最能区分好坏客户"的那个切分。衡量标准有两种：

**基尼系数（Gini Impurity）**

CART 决策树（sklearn 默认）使用基尼系数衡量节点纯度：

```
Gini = 1 - Σ p_k²

  p_k：节点中第 k 类样本的占比
```

- 节点全为坏客户或全为好客户 → Gini = 0，最纯
- 节点好坏各半 → Gini = 0.5，最混
- 每次分裂选择**加权平均 Gini 最小**的切分方式

**信息增益（Information Gain）**

ID3/C4.5 算法使用信息熵作为分裂标准：

```
熵（Entropy） = -Σ p_k × log₂(p_k)

信息增益 = 分裂前的熵 - 分裂后的加权平均熵
```

**实践选择**：风控建模通常使用 CART（基尼系数），sklearn 的 `DecisionTreeClassifier` 和 `RandomForestClassifier` 均默认使用 Gini。

### 风控参数推荐

**为什么风控决策树深度通常只有 2-3 层**

这是风控领域与通用机器学习最显著的差异之一：

| 出发点 | 说明 |
|--------|------|
| **监管可解释性** | 风控模型需向监管机构解释拒贷原因，超过 3 层的树规则已难以口头描述 |
| **业务可实施性** | 提炼出的规则需进入系统策略或人工审批流程，规则过复杂会增加执行成本 |
| **稳定性要求** | 层数越深，叶节点样本越少，规则越依赖训练数据的局部噪声，上线后稳定性差 |
| **防止过拟合** | 信贷数据好坏比通常 10:1 到 50:1，深树容易对稀有的坏样本过拟合 |

**经验规则**：
- 规则提炼用途：`max_depth = 2 或 3`，要求输出直接可用的业务策略
- 辅助分箱用途：`max_depth = 4 或 5`，稍深以找到更细致的切分点
- 特征初筛用途：`max_depth = 6–8`，允许稍深以充分捕捉特征关系

```python
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    max_depth=3,              # 树的最大深度（风控规则提炼通常 2-3）
    min_samples_split=100,    # 分裂节点所需的最小样本数
    min_samples_leaf=50,      # 叶节点最小样本数（防止规则建立在极少样本上）
    criterion='gini',
    class_weight='balanced',  # 自动按好坏比调整样本权重
    random_state=42
)
```

`min_samples_leaf` 的重要性：如果叶节点只有 20 个样本，其中 3 个坏客户，坏率 15%——但这个估计极不稳定，换一批数据可能变成 5% 或 30%。设置 `min_samples_leaf=50` 能保证每条规则有足够的统计支撑。

### 典型用法

**用法一：直接提炼业务规则**

决策树的分裂规则可直接转译为系统策略：

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

**用法二：辅助 WOE 分箱**

人工分箱容易主观，决策树可自动找到统计意义上最优的分割点，再结合业务经验微调：

```python
def find_optimal_splits(df, feature, target, max_depth=4):
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=50,
        class_weight='balanced'
    )
    dt.fit(df[[feature]], df[target])

    tree = dt.tree_
    thresholds = sorted(set(
        tree.threshold[tree.threshold != -2]  # -2 是叶节点标记
    ))
    print(f"{feature} 建议分割点: {[round(t, 2) for t in thresholds]}")
    return thresholds
```

**用法三：特征重要性初筛**

在进入 WOE 编码和逻辑回归建模前，快速排除明显无信息量的特征：

```python
import pandas as pd

importance_df = pd.DataFrame({
    'feature':    feature_names,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

useful_features = importance_df[importance_df['importance'] > 0.01]['feature'].tolist()
```

### 优缺点

| 优点 | 说明 |
|------|------|
| 规则直接可读 | 树结构可直接翻译为"如果…则…"规则，无需额外解释，监管友好 |
| 无需特征工程 | 不需要归一化，对异常值和缺失值鲁棒性较好 |
| 捕捉非线性关系 | 天然处理变量与违约率之间的非线性、非单调关系 |
| 无需分布假设 | 不要求变量服从正态分布 |
| 计算速度快 | 训练和预测都很快，适合快速迭代 |

| 缺点 | 说明 |
|------|------|
| 高方差、不稳定 | 训练数据稍有变化，树的结构可能大幅改变 |
| 单棵树精度有限 | 浅树（深度 2-3）的区分度通常不如集成模型 |
| 类别不平衡敏感 | 不设 class_weight 会导致规则偏向好客户 |
| 叶节点概率不稳定 | 样本少时违约率估计极不可靠，不适合直接作为评分输出 |

---

## 二、随机森林

### 核心原理

单棵决策树的高方差问题由随机森林（Random Forest）通过两个核心机制解决：

**Bootstrap 采样（Bagging）**

每棵树从原始训练集中有放回地随机抽取约 63% 的样本（约 37% 不被选中，即 OOB 样本）。不同的树见到不同的数据子集，相互之间的相关性降低，集成后方差显著减小。

**随机特征选择**

每个节点分裂时，不从全部特征中选最优切分，而是从随机抽取的 √n 个特征（n = 总特征数）中选择最优切分。即使存在强特征，弱特征也有机会参与构建，各棵树差异性更大，集成效果更好。

**预测机制**：

```
分类任务：多数投票
  → 超过半数的树预测为"坏客户"，则最终判为坏客户

概率输出：各树预测概率取平均
  → 更稳定的违约概率估计
```

### 核心参数推荐

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=300,         # 树的数量（越多越稳定，边际收益递减）
    max_depth=8,              # 用于精度时可比单棵决策树更深
    min_samples_leaf=30,      # 叶节点最小样本数
    max_features='sqrt',      # 每次分裂随机选取的特征数
    class_weight='balanced',  # 自动调整好坏样本权重
    oob_score=True,           # 开启 OOB 评估
    n_jobs=-1,
    random_state=42
)
```

| 参数 | 调优建议 |
|------|---------|
| n_estimators | 先设 100，若 OOB Score 仍在提升则加至 300–500 |
| max_depth | 通常 6–10；太深容易过拟合，太浅精度不足 |
| min_samples_leaf | 信贷数据建议 30–50 |
| max_features | 分类任务用 `'sqrt'`，特征极多时可尝试 `0.5` |

### OOB 评估

Bootstrap 采样使得每棵树约有 37% 的样本未被训练（OOB 样本），可直接用于评估，无需额外划分验证集：

```python
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

# OOB 预测概率（对每个样本，用没见过它的那些树来预测）
oob_proba = rf_model.oob_decision_function_[:, 1]

oob_auc = roc_auc_score(y_train, oob_proba)
ks_stat, _ = ks_2samp(oob_proba[y_train == 1], oob_proba[y_train == 0])

print(f"OOB AUC: {oob_auc:.4f}")
print(f"OOB KS:  {ks_stat:.4f}")
```

注意：OOB 样本混在了不同时间段，在信贷场景中不能完全替代按时间划分的测试集，但可作为快速参考。

### 特征筛选用法

在风控建模中，随机森林最常见的用途是**在进入 WOE 编码和逻辑回归建模前做特征预筛选**：

```python
rf_selector = RandomForestClassifier(
    n_estimators=200, max_depth=8,
    min_samples_leaf=30, class_weight='balanced',
    n_jobs=-1, random_state=42
)
rf_selector.fit(X_train, y_train)

importance_series = pd.Series(
    rf_selector.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

# 保留重要性前 50 名，再结合 IV 值做交叉验证
top_features = importance_series.head(50).index.tolist()
```

注意：随机森林特征重要性基于 Gini 下降量，对高基数（取值种类多）的连续型特征天然偏高。建议同时计算 IV 值，两个维度都低的特征再剔除。

### 优缺点

| 优点 | 说明 |
|------|------|
| 精度高、稳定 | 集成大量树后方差显著降低，对数据扰动不敏感 |
| 天然抗过拟合 | 随机采样 + 随机特征使各树互相独立，不容易同时过拟合 |
| 特征重要性可用 | 批量筛选数百候选特征时效率极高 |
| 处理高维特征 | 特征数量达数百乃至数千时仍能有效工作 |
| OOB 免费评估 | 无需额外划分验证集即可得到可靠的性能估计 |

| 缺点 | 说明 |
|------|------|
| 不可解释 | 数百棵树的集成无法直接提炼规则，不满足监管逐变量解释的要求 |
| 训练较慢 | 相比单棵决策树，训练时间显著增加（可并行缓解） |
| 内存占用大 | n_estimators=500 的模型存储可能达数 GB |
| 特征重要性有偏 | 对高基数特征偏高估，需结合 IV 值交叉验证 |

### 两者对比与风控标准工作流

| 维度 | 决策树（2-3 层） | 随机森林 |
|------|----------------|---------|
| 主要用途 | 规则提炼、分箱辅助 | 特征筛选、精度建模 |
| 区分度（KS） | 低（0.2–0.35） | 中高（0.35–0.5） |
| 可解释性 | 最高（规则直接可读） | 低（黑盒） |
| 稳定性 | 低（高方差） | 高（集成降低方差） |
| 监管友好度 | 高 | 低，不宜直接上线 |

**标准工作流**：

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

### 实战常见坑

1. **深度设太大做规则提炼**：`max_depth=10` 的决策树输出的规则复杂度堪比 XGBoost，深度 2-3 才是风控规则提炼的正确姿势

2. **忘记设 class_weight**：好坏 20:1 不设权重，模型把所有人预测为好客户，KS 接近 0

3. **用随机森林特征重要性直接排除变量**：重要性低不等于 IV 低，建议两个维度都低的特征再剔除

4. **随机森林直接当主模型上线**：强监管场景（银行、持牌消金）不宜直接作为评分模型，应作为特征筛选工具

5. **用 OOB Score 替代时序测试集**：OOB 样本混在不同时间段，无法替代按时间划分的真实测试集
