---
order: 9
title: "XGBoost 与 LightGBM：梯度提升在风控中的实战"
category: modeling
summary: 深入介绍梯度提升树的数学原理（目标函数、二阶泰勒展开）、XGBoost 与 LightGBM 的核心技术差异（GOSS、EFB、Leaf-wise）、风控专用参数配置（单调性约束、样本不平衡处理）、SHAP 可解释性全解，以及过拟合诊断与 Optuna 调参。
tags: [XGBoost, LightGBM, 梯度提升, GBDT, 风控建模, SHAP, 单调性约束, Optuna]
---

## 一、梯度提升的数学基础

### Boosting 的核心思想

XGBoost 和 LightGBM 都属于**梯度提升决策树（GBDT）**家族。理解它们之前，先看清楚 Boosting 和 Bagging 的根本区别：

**随机森林（Bagging）**：
```
多棵树独立并行训练 → 投票取平均
降低方差，但不降低偏差
```

**梯度提升（Boosting）**：
```
第 1 棵树：预测违约概率 → 有残差 r₁
第 2 棵树：专门拟合 r₁（纠错）→ 有残差 r₂
第 3 棵树：专门拟合 r₂ → ...
最终预测 = 所有树的输出加权叠加
降低偏差，但需要防止方差增大（过拟合）
```

### 目标函数与正则化

GBDT 在第 t 轮的优化目标是：

```
Obj(t) = Σ L(y_i, ŷ_i(t)) + Σ Ω(f_k)

  L         ：损失函数（二分类用对数损失）
  ŷ_i(t)    ：第 t 轮后第 i 个样本的预测值
  Ω(f_k)    ：第 k 棵树的复杂度惩罚项
```

**XGBoost 的正则化项**（这是 XGBoost 相较原始 GBDT 的关键创新）：

```
Ω(f) = γT + (1/2) λ Σ w_j²

  T   ：叶节点数量（惩罚叶节点过多）
  w_j ：第 j 个叶节点的权重（输出值）
  γ   ：新增叶节点的最小损失下降阈值（min_split_loss）
  λ   ：L2 正则化系数（reg_lambda）
```

### 二阶泰勒展开

XGBoost 使用**二阶泰勒展开**近似损失函数，相比原始 GBDT 仅使用一阶梯度，收敛更快、精度更高：

```
L(y, ŷ + f_t) ≈ L(y, ŷ) + g_i · f_t(x_i) + (1/2) h_i · f_t(x_i)²

  g_i = ∂L/∂ŷ    一阶梯度（gradient）
  h_i = ∂²L/∂ŷ²  二阶梯度（hessian）
```

对于二分类（对数损失）：
```
g_i = p_i - y_i       （预测概率 - 真实标签）
h_i = p_i(1 - p_i)    （预测概率的方差）
```

正是因为同时利用了梯度和曲率信息，XGBoost 在相同轮数下通常比传统 GBDT 精度更高。

---

## 二、XGBoost

### 核心技术创新

| 创新点 | 说明 |
|--------|------|
| **正则化目标函数** | 在目标函数中直接加入 L1/L2 正则和叶节点数惩罚，而非后处理剪枝 |
| **二阶泰勒展开** | 利用 Hessian 信息，比仅用梯度的 GBDT 收敛更快 |
| **加权分位数草图** | 大数据集上近似寻找最优分裂点，兼顾速度与精度 |
| **稀疏感知分裂** | 自动学习缺失值的最优分配方向，无需手动填充 |
| **列块并行** | 按特征维度并行计算分裂增益，不是树维度并行 |
| **Gamma 剪枝** | 新增节点的损失下降必须 > γ，否则不分裂，是额外的防过拟合手段 |

### 参数详解与风控推荐值

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    # ===== 树结构 =====
    n_estimators=1000,         # 配合 early_stopping 使用，设大一点
    max_depth=4,               # 风控通常 3–6；越深越容易过拟合
    min_child_weight=30,       # 叶节点最小 Hessian 和（等效最小样本数）
                               # 信贷场景建议 20–50，防止在少量坏样本上过拟合

    # ===== 学习速率 =====
    learning_rate=0.05,        # 越小越精细，配合更多 n_estimators

    # ===== 防过拟合：随机采样 =====
    subsample=0.8,             # 每轮随机抽取 80% 样本训练
    colsample_bytree=0.8,      # 每棵树随机使用 80% 特征
    colsample_bylevel=1.0,     # 每层随机使用的特征比例
    colsample_bynode=1.0,      # 每个节点随机使用的特征比例

    # ===== 防过拟合：正则化 =====
    reg_alpha=0.1,             # L1 正则（稀疏化系数）
    reg_lambda=1.0,            # L2 正则（收缩系数，默认 1）
    gamma=0.1,                 # 节点分裂所需的最小损失下降值
                               # > 0 时会更保守，减少叶节点数

    # ===== 样本不平衡 =====
    scale_pos_weight=20,       # 坏样本权重 = 好坏比（好:坏=20:1 则设 20）

    # ===== 单调性约束（风控核心！）=====
    # monotone_constraints 见下文专节

    # ===== 工程配置 =====
    tree_method='hist',        # 直方图算法，速度快，大数据推荐
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42
)
```

**参数优先级（调参顺序）**：

```
第一步：固定 learning_rate=0.05，用 early_stopping 确定 n_estimators
第二步：调 max_depth 和 min_child_weight（控制树的复杂度）
第三步：调 subsample 和 colsample_bytree（随机采样，防过拟合）
第四步：调 gamma、reg_alpha、reg_lambda（正则化）
第五步：适当降低 learning_rate，同步增加 n_estimators
```

### Early Stopping

Early Stopping 是防过拟合最有效的手段，应作为默认配置：

```python
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100
)

# 查看最佳轮数
print(f"最佳轮数: {model.best_iteration}")
print(f"最佳验证 AUC: {model.best_score}")
```

验证集 AUC 连续 50 轮不提升，训练自动停止，`model.best_iteration` 记录最优轮数。

### 单调性约束（风控的合规要求）

在信贷风控中，许多变量与违约率之间存在**必须单调**的关系——例如"逾期次数越多，违约概率应越高"。如果模型违背这一关系，不仅难以向监管解释，还可能是数据噪声导致的伪关系。

XGBoost 提供 `monotone_constraints` 参数强制约束：

```python
# 假设特征列表为：[逾期次数, 月收入, 贷款金额, 申请时长]
model = xgb.XGBClassifier(
    monotone_constraints={
        '近12月逾期次数': 1,   # +1 = 单调递增（逾期越多，违约概率越高）
        '月收入':         -1,  # -1 = 单调递减（收入越高，违约概率越低）
        '贷款金额':        1,   # +1 = 单调递增
        '申请时长':        0,   # 0  = 无约束
    },
    # 其他参数...
)
```

加入单调性约束后，模型在该特征上的预测曲线保证单调，牺牲少量精度换来更强的可解释性和合规性。

### 特征重要性的三种类型

XGBoost 提供三种特征重要性，含义不同，应结合使用：

```python
# weight：特征被用于分裂的次数（出现频率）
# gain：特征分裂带来的平均损失下降（精度贡献，最重要）
# cover：特征分裂覆盖的平均样本数

importance_gain  = model.get_booster().get_score(importance_type='gain')
importance_weight = model.get_booster().get_score(importance_type='weight')
importance_cover  = model.get_booster().get_score(importance_type='cover')

import pandas as pd
pd.DataFrame({
    'gain':  importance_gain,
    'weight': importance_weight,
    'cover':  importance_cover
}).sort_values('gain', ascending=False).head(20)
```

**推荐使用 `gain`**：weight 容易对高基数连续特征偏高（因为可能在很多不同阈值上分裂），gain 更准确反映特征对模型精度的实际贡献。

### XGBoost 的优缺点

| 优点 | 说明 |
|------|------|
| 精度高 | 二阶梯度 + 正则化目标函数，业界公认的高精度算法 |
| 正则化完备 | L1/L2/gamma 三重正则化，防过拟合手段丰富 |
| 缺失值自动处理 | 学习缺失值最优分配方向，无需手动填充 |
| 单调性约束 | 原生支持，满足信贷风控合规要求 |
| 可解释性（SHAP） | 配合 SHAP 可提供个体级别的特征贡献分解 |
| 成熟稳定 | 2014 年发布，社区生态完善，经过大量生产验证 |

| 缺点 | 说明 |
|------|------|
| 大数据较慢 | Level-wise 生长策略在百万级以上数据集上速度不及 LightGBM |
| 内存占用高 | 按列存储需要较大内存，特征数多时更明显 |
| 类别特征需编码 | 不支持原生类别特征，需手动 WOE/Label Encoding |
| 调参复杂 | 参数数量多，调参经验要求高 |

---

## 三、LightGBM

### 核心技术创新

LightGBM（Light Gradient Boosting Machine）是微软 2017 年发布的改进版 GBDT，专为大数据和高维特征场景设计，在速度和内存上显著优于 XGBoost。

**Leaf-wise 生长策略（vs XGBoost 的 Level-wise）**：

```
XGBoost（Level-wise，按层生长）：
  每轮分裂同一层的所有节点
  → 树的形状规整，每层宽度相等
  → 不容易过拟合，但精度提升有限

LightGBM（Leaf-wise，按叶生长）：
  每轮只分裂当前损失下降最大的那个叶节点
  → 同样叶节点数下，树更深且不对称
  → 精度更高，但更容易过拟合（需严格控制 num_leaves）
```

**GOSS：梯度单边采样**

传统 GBDT 每轮都要遍历所有样本。LightGBM 的 GOSS（Gradient-based One-Side Sampling）策略：
- 保留**梯度绝对值大**的样本（这些样本误差大，是重点训练对象）
- 对**梯度绝对值小**的样本随机抽取一个小子集
- 结合两部分计算信息增益

结果：减少 80%+ 的样本计算量，精度损失极小。

**EFB：互斥特征捆绑**

高维稀疏数据（如行为特征）中，很多特征互斥（不同时非零）。EFB（Exclusive Feature Bundling）将互斥特征合并为一个特征：
- 减少特征数量
- 降低内存占用
- 训练速度大幅提升

### 参数详解与风控推荐值

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    # ===== 树结构 =====
    n_estimators=1000,          # 配合 early_stopping 使用
    num_leaves=31,              # 核心参数！每棵树最大叶节点数
                                # 建议 < 2^max_depth，通常 15–63
                                # 不要超过 63，否则极易过拟合
    max_depth=-1,               # -1 不限深度，用 num_leaves 控制复杂度
    min_child_samples=50,       # 叶节点最小样本数（等效 min_data_in_leaf）
                                # 信贷场景建议 30–100

    # ===== 学习速率 =====
    learning_rate=0.05,

    # ===== 防过拟合：采样 =====
    subsample=0.8,              # 等效 bagging_fraction
    subsample_freq=5,           # 每 5 轮执行一次 bagging（需 > 0 才生效）
    colsample_bytree=0.8,       # 等效 feature_fraction

    # ===== 正则化 =====
    reg_alpha=0.1,              # L1 正则
    reg_lambda=1.0,             # L2 正则
    min_split_gain=0.0,         # 分裂所需的最小增益（等效 gamma）

    # ===== 样本不平衡 =====
    class_weight='balanced',    # 或手动设 scale_pos_weight
    # is_unbalance=True,        # LightGBM 专有：自动调整权重

    # ===== 类别特征（LightGBM 原生支持）=====
    # categorical_feature=['province', 'occupation'],

    # ===== 单调性约束 =====
    # monotone_constraints=[1, -1, 0, ...]  # 与特征顺序对应

    # ===== 工程 =====
    n_jobs=-1,
    random_state=42,
    verbose=-1                  # 关闭训练日志
)
```

**num_leaves 与 max_depth 的关系**：

```
若 max_depth = 6，对称树最多有 2^6 = 64 个叶节点
LightGBM Leaf-wise 生长时，建议 num_leaves < 2^max_depth

实践建议：
  小数据集（< 10 万）：num_leaves = 15–31
  中等数据集（10–100 万）：num_leaves = 31–63
  大数据集（> 100 万）：num_leaves = 63–127，但需加大 min_child_samples
```

### 类别特征原生支持

LightGBM 可以直接使用类别特征，无需 One-hot 编码，内部使用最优分组策略：

```python
model = lgb.LGBMClassifier(...)
model.fit(
    X_train, y_train,
    categorical_feature=['province', 'occupation', 'education'],
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
```

相比 One-hot 展开，原生类别特征处理更快，且能找到更优的分组组合。

### LightGBM 的优缺点

| 优点 | 说明 |
|------|------|
| 训练速度快 | GOSS + EFB，大数据集比 XGBoost 快 3–10 倍 |
| 内存占用低 | 直方图算法 + EFB，内存是 XGBoost 的 1/3 左右 |
| 原生类别特征 | 无需 One-hot 编码，自动找最优分组 |
| 高维特征表现好 | EFB 对稀疏高维特征特别有效 |
| 精度与 XGBoost 相当或略高 | Leaf-wise 在同等叶节点下精度更高 |

| 缺点 | 说明 |
|------|------|
| 过拟合风险更高 | Leaf-wise 策略在小数据集上更容易过拟合 |
| num_leaves 调参难 | 设置不当直接导致过拟合，比 max_depth 更难直觉理解 |
| 小数据集表现不稳定 | GOSS 采样在样本少时可能引入额外噪声 |
| 类别特征支持有限 | 高基数类别（数百种取值）处理效果不如 target encoding |

---

## 四、XGBoost vs LightGBM 对比

| 维度 | XGBoost | LightGBM |
|------|---------|---------|
| 生长策略 | Level-wise（按层） | Leaf-wise（按叶） |
| 训练速度 | 快 | 更快（大数据优势显著） |
| 内存占用 | 较高 | 低 |
| 精度 | 高 | 相当或略高 |
| 过拟合风险 | 中 | 较高（需仔细控制 num_leaves） |
| 类别特征 | 需手动编码 | 原生支持 |
| 单调性约束 | 原生支持 | 原生支持 |
| 成熟度 | 高（2014） | 高（2017） |
| 适用场景 | 中小数据、精度敏感 | 大数据、高维特征 |

**如何选择**：
- 数据量 < 50 万行：两者速度差异不大，优先 XGBoost（更稳定）
- 数据量 > 100 万行，或特征数 > 500：优先 LightGBM
- 强调单调性合规的评分卡：两者均支持，XGBoost 生态更成熟

---

## 五、风控建模全流程

### 数据切分（按时间，不能随机）

```python
# 按时间切分，防止未来信息泄露
df_train = df[df['apply_month'] <= '2023-06']
df_val   = df[(df['apply_month'] > '2023-06') & (df['apply_month'] <= '2023-09')]
df_test  = df[df['apply_month'] > '2023-09']

# 验证集必须比训练集更新，early_stopping 才有意义
```

### 训练与调参

```python
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=30,
    scale_pos_weight=20,
    gamma=0.1,
    reg_lambda=1.0,
    tree_method='hist',
    early_stopping_rounds=50,
    eval_metric='auc',
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

# 计算 AUC 和 KS
test_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, test_proba)
ks, _ = ks_2samp(test_proba[y_test == 1], test_proba[y_test == 0])
print(f"Test AUC: {auc:.4f}  KS: {ks:.4f}")
```

### 用 Optuna 做贝叶斯调参

手动网格搜索效率低下，Optuna 使用贝叶斯优化自动搜索参数空间：

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'n_estimators': 1000,
        'max_depth':         trial.suggest_int('max_depth', 3, 7),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'min_child_weight':  trial.suggest_int('min_child_weight', 10, 100),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma':             trial.suggest_float('gamma', 0, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0, 2.0),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.5, 3.0),
        'scale_pos_weight':  20,
        'tree_method': 'hist',
        'early_stopping_rounds': 50,
        'eval_metric': 'auc',
        'random_state': 42,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)
    val_proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, val_proba)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"最优参数: {study.best_params}")
print(f"最优验证 AUC: {study.best_value:.4f}")
```

---

## 六、SHAP：让模型决策可解释

### SHAP 的理论基础

SHAP（SHapley Additive exPlanations）来源于博弈论中的 Shapley 值，其核心思想是：**每个特征对最终预测值的贡献，等于它在所有可能的特征子集组合中带来的边际贡献的加权平均**。

```
f(x) = E[f(x)] + Σ φ_i

  E[f(x)] ：模型的基准预测（所有样本的平均预测值）
  φ_i     ：特征 i 的 SHAP 值（对预测值的贡献）
  所有 SHAP 值加总 = 预测值与基准值之差
```

SHAP 具有以下保证：
- **一致性**：特征贡献越大，SHAP 值绝对值越大
- **局部准确性**：所有特征 SHAP 值之和 = 预测值 - 基准值
- **缺失特征为 0**：模型中未使用的特征 SHAP 值为 0

### 全局特征重要性

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 特征重要性条形图（平均 |SHAP| 值）
shap.summary_plot(shap_values, X_test, plot_type='bar')

# 蜂巢图（同时展示特征值与 SHAP 值的关系）
# 颜色代表特征值高低，横轴为 SHAP 值正负
shap.summary_plot(shap_values, X_test)
```

![SHAP 蜂巢图示例](/images/shap-beeswarm.png)

**蜂巢图解读**：
- 每个点代表一个样本
- 横轴位置 = 该样本该特征的 SHAP 值（正值推高违约概率，负值降低）
- 颜色 = 该特征的原始值（红色高，蓝色低）
- 若红色点集中在右侧 → 该特征值越高，违约概率越高（如逾期次数）

### 单样本解释（解释具体一笔拒贷原因）

```python
# Waterfall 图：展示单个样本各特征对预测值的贡献
sample_idx = 5
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[sample_idx],
        feature_names=feature_names
    )
)
# 从基准值出发，每个特征的 SHAP 值使预测值向上（红）或向下（蓝）移动
# 最终累加到该样本的实际预测值

# Force 图（交互式，适合展示给业务人员）
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_idx],
    X_test.iloc[sample_idx],
    feature_names=feature_names
)
```

### 特征依赖图（Dependence Plot）

```python
# 查看某特征的 SHAP 值随特征值的变化关系
# 可发现非线性、阈值效应
shap.dependence_plot(
    '近12月逾期次数',   # 主特征
    shap_values,
    X_test,
    interaction_index='负债收入比'  # 着色特征（展示交互效应）
)
```

![SHAP 特征依赖图示例](/images/shap-dependence.png)

**依赖图解读**（以上图为例）：
- **左图（负债收入比）**：SHAP 值随负债收入比升高而增大，说明负债越重违约风险越高；颜色（月收入）偏蓝的点（低收入）集中在图的上方，揭示低收入 + 高负债的交互效应最强
- **右图（近12月逾期次数）**：SHAP 值在逾期 0 次时为负（降低违约概率），逾期 1 次以上迅速跳升，呈现明显的阶梯效应；着色（近3月申请次数）红色点偏上，说明逾期 + 高频申请的叠加风险更高

### SHAP 在风控合规中的应用

| 场景 | 用法 |
|------|------|
| 解释拒贷原因 | Waterfall/Force 图展示哪些特征导致该申请人被拒 |
| 监管报告 | Summary Plot 展示全局变量重要性顺序 |
| 模型诊断 | PSI 超标时，对比不同时期样本的 SHAP 值分布，定位漂移特征 |
| 策略优化 | Dependence Plot 找到变量的风险阈值，辅助制定策略切点 |

---

## 七、过拟合的诊断与处理

### 学习曲线诊断

```python
results = model.evals_result()
import matplotlib.pyplot as plt

train_auc = results['validation_0']['auc']
val_auc   = results['validation_1']['auc']
epochs    = range(len(train_auc))

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_auc, label='Train AUC')
plt.plot(epochs, val_auc,   label='Val AUC')
plt.axvline(model.best_iteration, color='r', linestyle='--', label='Best Iteration')
plt.xlabel('Boosting Round')
plt.ylabel('AUC')
plt.legend()
plt.show()
```

**四种典型形态**：

| 形态 | 特征 | 诊断 |
|------|------|------|
| 正常 | 两条曲线同步上升，最终趋于平稳 | 正常，模型健康 |
| 过拟合 | Train 持续上升，Val 先升后降 | 需减少复杂度或加强正则 |
| 欠拟合 | 两条曲线都低且平坦 | 模型太简单，增加 n_estimators 或加深树 |
| 数据泄露 | Train AUC = 1.0 或接近 | 特征中含有目标变量相关信息，立即排查 |

### 过拟合处理优先级

| 方案 | 操作 | 效果 |
|------|------|------|
| Early Stopping | 根据验证集自动停止 | 首选，最有效 |
| 降低 max_depth | 从 6 降到 4 或 3 | 高效 |
| 增大 min_child_weight | 从 10 增到 50 | 高效 |
| 降低 num_leaves（LightGBM） | 从 63 降到 31 | 高效 |
| 降低 learning_rate | 同步增加 n_estimators | 中等，需更长训练时间 |
| 增大正则化（alpha/lambda） | L1/L2 正则 | 中等 |
| 减少特征数 | 去掉低重要性特征 | 中等 |
| 增大 subsample | 每轮采样更少样本 | 轻微 |

---

## 八、三类模型在风控中的定位

| 维度 | 逻辑回归 | 随机森林 | XGBoost/LightGBM |
|------|---------|---------|-----------------|
| 核心原理 | 线性模型 + Sigmoid | Bagging + 投票 | Boosting + 梯度纠错 |
| 预测精度 | 中 | 中高 | 最高 |
| 可解释性 | 最高（系数直读） | 中（特征重要性） | 低（需 SHAP 辅助） |
| 特征工程要求 | 高（需 WOE 等） | 低 | 中 |
| 训练速度 | 最快 | 中 | 快（LightGBM 最快） |
| 过拟合风险 | 低 | 低 | 高（需仔细控制） |
| 监管友好度 | 最高 | 中 | 需配合 SHAP |

**典型建模组合**：

```
组合 1（传统合规路线）：
  随机森林（特征筛选）→ WOE 分箱 → 逻辑回归评分卡
  优点：监管友好，可解释性最强
  适用：申请评分卡、银行客群

组合 2（精度优先路线）：
  LightGBM（主模型）+ SHAP（解释）+ PSI 监控
  优点：精度最高，灵活
  适用：互联网消费金融、实时风控

组合 3（双模型互验）：
  逻辑回归（基准线）+ XGBoost（精度提升）
  两个模型 KS 差距大时，检查是否有数据泄露
```

---

## 九、实战常见坑

1. **时序泄露**：用未来数据的特征（如"30 天后的还款行为"）预测违约，线下好线上差

2. **目标泄露**：特征和标签来自同一时间点，如用"当月逾期天数"预测"是否逾期"，AUC 接近 1

3. **num_leaves 设太大（LightGBM）**：`num_leaves=127` 配合小数据集必然严重过拟合，生产环境通常不超过 63

4. **忘记设 scale_pos_weight**：好坏比 20:1 不设权重，模型把所有人预测为好客户

5. **验证集不够新**：验证集应时间上晚于训练集，否则 Early Stopping 失去意义

6. **单调性约束遗漏**：风控模型上线后被质疑"收入高的客户违约率反而更高"，这类情况应在建模时用 monotone_constraints 规避

7. **SHAP 基准值误解**：`explainer.expected_value` 是训练/背景数据的平均预测概率，不是 50%；展示给业务时需说明基准含义
