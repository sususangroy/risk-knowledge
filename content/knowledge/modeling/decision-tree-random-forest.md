---
title: "决策树与随机森林：从规则提炼到集成学习"
category: modeling
summary: 介绍决策树的核心原理（分裂准则、剪枝）、随机森林的集成思想，以及它们在信贷风控中的典型用法——特征重要性筛选、策略规则提炼、与逻辑回归的协同。
tags: [决策树, 随机森林, 集成学习, 风控建模, 特征重要性]
---

## 一、决策树：像人一样做决策

决策树（Decision Tree）是最直观的机器学习算法。它的逻辑和人类做信贷决策的方式高度相似：

```
是否批准贷款？
├── 年龄 < 22 岁？
│   ├── 是 → 拒绝（高风险）
│   └── 否 → 继续判断
│       ├── 历史逾期次数 > 2 次？
│       │   ├── 是 → 拒绝
│       │   └── 否 → 月收入 > 5000？
│       │       ├── 是 → 通过
│       │       └── 否 → 拒绝
```

每一个"问题"就是一个节点（Node），最终的结论（通过/拒绝）是叶节点（Leaf）。

---

## 二、决策树是怎么"学习"分裂规则的

### 核心问题：选哪个特征、在哪里切

树模型在训练时，会自动从所有特征和所有可能的切分点中，找到"最能区分好坏客户"的那个切分。

### 基尼系数（Gini Impurity）

CART 决策树（sklearn 默认）使用**基尼系数**来衡量纯度：

```
Gini = 1 - Σ pᵢ²

其中 pᵢ 是每个类别的比例
```

**直觉理解**：
- 一个节点里全是坏客户（或全是好客户）→ Gini = 0，最纯
- 一个节点里好坏各半 → Gini = 0.5，最混
- 选择能让**加权平均 Gini 最小**的分裂方式

### 信息增益（ID3/C4.5 使用）

另一种分裂标准，基于信息熵：

```
信息增益 = 分裂前的熵 - 分裂后的加权平均熵
```

选择信息增益最大的特征进行分裂。

**实际选择**：风控建模中 CART（基尼系数）更常用，随机森林和 XGBoost 也默认使用 CART。

---

## 三、防止过拟合：剪枝

未经限制的决策树会无限生长，最终每个叶节点只有一两个样本——在训练集完美，在测试集一塌糊涂。

### 预剪枝（Pre-pruning）：生长时限制

| 参数 | 含义 | 建议值（信贷场景） |
|------|------|-----------------|
| max_depth | 树的最大深度 | 3–6 层 |
| min_samples_split | 分裂所需最小样本数 | 50–200 |
| min_samples_leaf | 叶节点最小样本数 | 20–100 |
| max_features | 每次分裂考虑的特征数 | sqrt(n) 或 0.7 |

### 后剪枝（Post-pruning）：长完再裁

先让树完全生长，再从叶节点开始删减提升最少的分支。scikit-learn 的 `ccp_alpha` 参数控制后剪枝强度。

---

## 四、决策树在风控中的典型用法

### 用法一：规则提炼工具

决策树输出的分裂规则可以直接转为业务规则，方便向业务团队解释：

```python
from sklearn.tree import export_text

rules = export_text(dt_model, feature_names=feature_names)
print(rules)
# 输出类似：
# |--- 历史逾期次数 <= 1.50
# |   |--- 月收入 > 3000.00
# |   |   |--- class: 好客户
# |   |--- 月收入 <= 3000.00
# |   |   |--- class: 坏客户
```

### 用法二：特征重要性初筛

```python
import pandas as pd
import matplotlib.pyplot as plt

# 获取特征重要性
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

# 可视化
importance_df.head(20).plot(kind='barh', x='feature', y='importance')
plt.title('决策树特征重要性')
plt.show()
```

### 用法三：分箱辅助

利用决策树自动找到最优分割点，辅助逻辑回归的 WOE 分箱，比人工分箱更客观。

---

## 五、随机森林：多棵树的集体智慧

单棵决策树的问题在于**方差高**——训练数据稍有变化，树的结构就会大幅改变，预测结果不稳定。

随机森林（Random Forest）用两个"随机"解决这个问题：

### 两个关键随机性

**1. Bootstrap 采样（Bagging）**：

每棵树使用原始训练数据的**随机子集**（约 63%）训练，不同的树见过不同的数据，降低相互依赖。

**2. 随机特征选择**：

每个节点分裂时，只从**随机选出的部分特征**中选择最优分裂，通常取 `sqrt(特征总数)` 个。这使得不同的树具有差异性，避免强特征主导所有树。

### 预测：投票/平均

```
分类任务：多数投票（超过一半的树说是坏客户 → 判为坏客户）
回归任务：平均所有树的预测值
```

### 随机森林 vs 单棵决策树

| 维度 | 单棵决策树 | 随机森林 |
|------|-----------|---------|
| 稳定性 | 低，容易过拟合 | 高 |
| 预测精度 | 中 | 高 |
| 可解释性 | 很高（可直接读规则） | 中（特征重要性，但规则不透明） |
| 训练速度 | 快 | 较慢（需训练 N 棵树） |
| 适合场景 | 规则提炼、可解释要求高 | 特征筛选、精度要求较高 |

---

## 六、随机森林的核心参数

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,        # 树的数量，越多越稳定，通常 100-500
    max_depth=8,             # 每棵树最大深度
    min_samples_leaf=50,     # 叶节点最小样本数（防过拟合）
    max_features='sqrt',     # 每次分裂的特征数
    class_weight='balanced', # 处理样本不平衡
    n_jobs=-1,               # 并行训练，-1 使用所有 CPU
    random_state=42
)
```

**关键调参建议**：

1. `n_estimators`：先用 100，看精度不够再加，边际效益递减
2. `max_depth`：控制在 6–10，太深容易过拟合
3. `min_samples_leaf`：信贷数据至少 30–50，防止小节点过拟合
4. `max_features`：分类任务用 'sqrt'，回归任务用 'auto'

---

## 七、OOB 评估：免费的验证集

随机森林训练时，每棵树大约有 37% 的数据**没被采样到**（Out-Of-Bag，OOB）。

可以用 OOB 数据评估模型，相当于一个免费的验证集：

```python
rf_model = RandomForestClassifier(
    n_estimators=200,
    oob_score=True,  # 开启 OOB 评估
    random_state=42
)
rf_model.fit(X_train, y_train)

print(f"OOB Score: {rf_model.oob_score_:.4f}")
```

---

## 八、风控实战建议

### 用随机森林做特征筛选

在做逻辑回归或 XGBoost 建模前，先用随机森林跑一遍特征重要性，去掉重要性极低的特征，再细化 WOE 分箱。

```python
# 筛选重要性 top 30 的特征
importance_series = pd.Series(
    rf_model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

top_features = importance_series.head(30).index.tolist()
```

### 稳定性 vs 精度的权衡

在信贷风控中，模型稳定性往往比精度更重要：
- 一个 KS=0.45 但 PSI 长期 < 0.1 的模型，比 KS=0.52 但两个月后 PSI > 0.25 的模型更有价值
- 随机森林通常比单棵决策树稳定得多，也比 XGBoost 稳定（代价是精度略低）

### 不适合随机森林的场景

- **强监管要求可解释**：需要逐变量解释系数时，随机森林不合适
- **特征极多（1000+）**：训练时间可能较长，考虑用 LightGBM
- **数据量极小（< 1000 样本）**：Bagging 效果有限
