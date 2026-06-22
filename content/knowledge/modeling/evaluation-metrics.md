---
title: "风控建模核心评价指标：WOE、IV、KS、AUC、PSI 全解"
category: modeling
summary: 系统介绍信贷风控建模中最常用的评价指标——WOE/IV（特征筛选）、KS/AUC（模型区分度）、PSI（稳定性监控）——包含数学定义、直觉解释、计算代码和判断标准。
tags: [WOE, IV, KS, AUC, PSI, 模型评估, 风控建模, 评分卡]
---

## 一、指标体系概览

风控建模中的评价指标按用途分为三类：

| 类别 | 指标 | 核心问题 |
|------|------|----------|
| **特征筛选** | WOE、IV | 这个变量有没有预测能力？ |
| **模型区分度** | KS、AUC / Gini 系数 | 模型能不能把好坏客户分开？ |
| **模型稳定性** | PSI、CSI | 模型上线后还有效吗？ |

---

## 二、WOE：证据权重

### 定义

WOE（Weight of Evidence，证据权重）衡量某个特征的某个分箱，相比整体而言更倾向于"好客户"还是"坏客户"。

```
WOE_i = ln( 坏客户占比_i / 好客户占比_i )
      = ln( (B_i / B_total) / (G_i / G_total) )

其中：
  B_i     ：第 i 个分箱内的坏客户数
  G_i     ：第 i 个分箱内的好客户数
  B_total ：全体坏客户数
  G_total ：全体好客户数
```

### 直觉解释

| WOE 值 | 含义 |
|--------|------|
| WOE > 0（正值） | 该分箱坏客户比例**高于**整体均值，风险偏高 |
| WOE = 0 | 该分箱坏客户比例与整体相同，无区分能力 |
| WOE < 0（负值） | 该分箱坏客户比例**低于**整体均值，风险偏低 |
| 绝对值越大 | 区分能力越强 |

**举例**：历史逾期次数分箱后：
```
逾期 0 次：WOE = -0.8（好客户集中区，风险低）
逾期 1 次：WOE = +0.3（略高风险）
逾期 2+ 次：WOE = +1.5（坏客户集中区，风险极高）
```

### 计算代码

```python
import numpy as np
import pandas as pd

def calc_woe_iv(df, feature, target, bins=10):
    total_bad  = df[target].sum()
    total_good = (df[target] == 0).sum()

    df['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')

    grouped = df.groupby('bin')[target].agg(['sum', 'count'])
    grouped.columns = ['bad', 'total']
    grouped['good'] = grouped['total'] - grouped['bad']

    grouped['bad_rate']  = grouped['bad']  / total_bad
    grouped['good_rate'] = grouped['good'] / total_good

    # 防止除零
    grouped['bad_rate']  = grouped['bad_rate'].replace(0, 0.0001)
    grouped['good_rate'] = grouped['good_rate'].replace(0, 0.0001)

    grouped['woe'] = np.log(grouped['bad_rate'] / grouped['good_rate'])
    grouped['iv']  = (grouped['bad_rate'] - grouped['good_rate']) * grouped['woe']

    return grouped[['bad', 'good', 'woe', 'iv']], grouped['iv'].sum()
```

### WOE 编码在逻辑回归中的作用

将原始特征替换为对应分箱的 WOE 值后：
1. 逻辑回归可以捕捉**非线性关系**（因为每个分箱有独立 WOE）
2. 模型系数变得**可比较**（不同量纲的变量统一转为 WOE 后可比）
3. 缺失值单独作为一个分箱，有其 WOE 值，**无需额外填充**

---

## 三、IV：信息值

### 定义

IV（Information Value，信息值）汇总了某个特征所有分箱的 WOE，衡量该特征的**整体预测能力**。

```
IV = Σ (坏客户占比_i - 好客户占比_i) × WOE_i
   = Σ (B_i/B_total - G_i/G_total) × WOE_i
```

IV 是 WOE 的加权求和：分箱中好坏客户差距越大，WOE 绝对值越大，对 IV 的贡献越多。

### 判断标准

| IV 值范围 | 预测能力 | 操作建议 |
|-----------|---------|---------|
| < 0.02 | 几乎无预测能力 | 直接丢弃 |
| 0.02 – 0.1 | 弱预测能力 | 谨慎使用 |
| 0.1 – 0.3 | 中等预测能力 | 可入模 |
| 0.3 – 0.5 | 强预测能力 | 优先入模 |
| > 0.5 | 极强，异常 | **务必检查数据泄露** |

> **注意**：IV > 0.5 几乎不可能在干净的信贷数据中出现。如果出现，大概率是用了"未来信息"——例如用"是否当月逾期"去预测"是否逾期"，IV 会无限接近完美。

### IV 用于特征筛选的流程

```python
# 批量计算所有特征的 IV
iv_results = {}
for feature in feature_list:
    try:
        _, iv = calc_woe_iv(df_train, feature, 'is_bad')
        iv_results[feature] = iv
    except Exception:
        iv_results[feature] = 0

iv_series = pd.Series(iv_results).sort_values(ascending=False)

# 筛选 IV > 0.02 的特征
selected_features = iv_series[iv_series > 0.02].index.tolist()
print(f"从 {len(feature_list)} 个特征中筛出 {len(selected_features)} 个")
```

---

## 四、KS：好坏客户的最大分离度

### 定义

KS（Kolmogorov-Smirnov 统计量）衡量模型在最优阈值处，能把好坏客户分开多远。

```
KS = max | 坏客户累积占比(x) - 好客户累积占比(x) |

其中 x 遍历所有可能的评分阈值
```

### 直觉理解

把所有客户按模型给出的违约概率从低到高排序，然后分别画出好客户和坏客户的**累积占比曲线**：

```
累积占比
100% |                          ________坏客户曲线
     |                ___-------
     |        ___-----
     |  ___---
  0% |__________________________
     低风险评分 ←————→ 高风险评分

         ↑ 好客户曲线（在坏客户曲线下方）

KS = 两条曲线之间的最大竖直距离
```

**理想情况**：高风险区坏客户集中，低风险区好客户集中，两条曲线分得很开，KS 大。

### 判断标准

| KS 值 | 评价 |
|-------|------|
| < 0.2 | 较差，模型基本无效 |
| 0.2 – 0.3 | 一般，勉强可用 |
| 0.3 – 0.45 | 良好，达到上线标准 |
| 0.45 – 0.6 | 优秀 |
| > 0.6 | 极好，需警惕过拟合或数据泄露 |

### 计算代码

```python
from scipy.stats import ks_2samp

def calc_ks(y_true, y_pred_proba):
    bad_scores  = y_pred_proba[y_true == 1]
    good_scores = y_pred_proba[y_true == 0]
    ks_stat, _ = ks_2samp(bad_scores, good_scores)
    return ks_stat

# 分档计算（更直观，可见每档的好坏分布）
def ks_table(y_true, y_pred_proba, n_bins=10):
    df = pd.DataFrame({'score': y_pred_proba, 'label': y_true})
    df['bin'] = pd.qcut(df['score'], q=n_bins, duplicates='drop', labels=False)

    total_bad  = y_true.sum()
    total_good = (y_true == 0).sum()

    table = df.groupby('bin')['label'].agg(['sum', 'count'])
    table.columns = ['bad', 'total']
    table['good']          = table['total'] - table['bad']
    table['cum_bad_rate']  = table['bad'].cumsum()  / total_bad
    table['cum_good_rate'] = table['good'].cumsum() / total_good
    table['ks']            = abs(table['cum_bad_rate'] - table['cum_good_rate'])

    return table
```

---

## 五、AUC 与 ROC 曲线

### 定义

**ROC 曲线**（Receiver Operating Characteristic）以**假正例率（FPR）**为横轴、**真正例率（TPR）**为纵轴，在所有阈值下描绘模型的分类能力：

```
TPR（召回率）= TP / (TP + FN)   坏客户被正确识别的比例
FPR（误拒率）= FP / (FP + TN)   好客户被误拒的比例
```

**AUC**（Area Under the ROC Curve）是 ROC 曲线下方的面积。

### 概率解释

```
AUC = P(模型给随机一个坏客户的评分 > 给随机一个好客户的评分)
```

即：**随机抽一对好客户和坏客户，模型正确区分其顺序的概率**。

| AUC 值 | 含义 |
|--------|------|
| 0.5 | 随机猜测，模型无效 |
| 0.6 – 0.7 | 一般 |
| 0.7 – 0.8 | 良好 |
| 0.8 – 0.9 | 优秀 |
| > 0.9 | 极好，警惕泄露 |

### KS 与 AUC 的关系

两者都衡量区分度，近似关系为：

```
KS ≈ 2 × (AUC - 0.5)
```

**选哪个？**

| 场景 | 推荐指标 |
|------|---------|
| 风控报告、上线标准 | KS（行业更通用，直觉更强） |
| 模型调参、学术对比 | AUC（数学性质更好，不受分箱影响） |
| 两者都汇报最佳 | 结合使用 |

### 计算代码

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc:.4f}")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate（误拒率）')
plt.ylabel('True Positive Rate（坏客户识别率）')
plt.legend()
plt.show()
```

### Gini 系数（信贷语境）

信贷领域有时用 **Gini 系数**而非 AUC：

```
Gini = 2 × AUC - 1
```

这与决策树中的"基尼不纯度"是**完全不同的概念**，仅仅是 AUC 的线性变换，范围为 [0, 1]。

---

## 六、PSI：群体稳定性指数

### 定义

PSI（Population Stability Index，群体稳定性指数）监控模型上线后，**评分分布是否发生偏移**。

```
PSI = Σ (A_i - E_i) × ln(A_i / E_i)

其中：
  E_i ：建模时第 i 分段的客户占比（预期分布）
  A_i ：上线后某月第 i 分段的客户占比（实际分布）
```

PSI 本质上是 KL 散度的对称化版本，衡量两个分布之间的差距。

### 判断标准

| PSI 值 | 含义 | 操作建议 |
|--------|------|---------|
| < 0.1 | 分布稳定 | 正常，无需动作 |
| 0.1 – 0.25 | 轻微偏移 | 需关注，排查原因 |
| > 0.25 | 显著偏移 | 模型可能失效，考虑重训 |

### 计算代码

```python
def calc_psi(expected_scores, actual_scores, n_bins=10):
    breakpoints = np.percentile(expected_scores, np.linspace(0, 100, n_bins + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    def get_rates(scores):
        counts = np.histogram(scores, bins=breakpoints)[0]
        rates  = counts / len(scores)
        return np.where(rates == 0, 0.0001, rates)

    expected_rates = get_rates(expected_scores)
    actual_rates   = get_rates(actual_scores)

    return np.sum((actual_rates - expected_rates) * np.log(actual_rates / expected_rates))

# 月度监控
for month in production_months:
    actual = df_prod[df_prod['month'] == month]['score'].values
    psi = calc_psi(train_scores, actual)
    print(f"{month}: PSI = {psi:.4f} {'偏移' if psi > 0.1 else '稳定'}")
```

### PSI 偏移的常见原因

| 原因类型 | 具体表现 |
|----------|---------|
| 客群结构变化 | 拓展新渠道、新产品，申请人特征与建模期不同 |
| 宏观经济冲击 | 疫情、经济下行导致整体信用水平下移 |
| 特征数据变化 | 上游数据源断供、字段口径改变 |
| 季节性波动 | 年末/年初申请人群体结构周期性变化 |

---

## 七、CSI：特征稳定性指数

### 定义

PSI 监控**评分整体分布**，而 CSI（Characteristic Stability Index）监控**单个特征分布**，用于定位 PSI 偏移的根因。

公式与 PSI 完全相同，只是把"评分分布"换成"某个特征的分布"：

```
CSI_j = Σ (A_i - E_i) × ln(A_i / E_i)   （针对特征 j 的各分箱）
```

当 PSI 超标时，逐一计算所有入模特征的 CSI，找出 CSI 最高的特征，即可定位是哪个特征发生了分布漂移，进而排查数据问题或业务变化。

```python
csi_results = {}
for feature in model_features:
    expected = df_train[feature].values
    actual   = df_prod_latest[feature].values
    csi_results[feature] = calc_psi(expected, actual)

csi_series = pd.Series(csi_results).sort_values(ascending=False)
print("CSI 最高的特征（可能的漂移原因）：")
print(csi_series.head(10))
```

---

## 八、VIF：多重共线性检验

### 定义

VIF（Variance Inflation Factor，方差膨胀因子）检测多个特征之间是否存在**多重共线性**，这会导致逻辑回归系数不稳定（符号错误、数值极大/极小）。

```
VIF_j = 1 / (1 - R²_j)

其中 R²_j 是将特征 j 作为因变量、其余特征作为自变量进行回归时的决定系数
```

| VIF 值 | 含义 |
|--------|------|
| 1 | 与其他特征完全无相关 |
| 1 – 5 | 轻微相关，可接受 |
| 5 – 10 | 中度共线，需关注 |
| > 10 | 严重共线，需处理 |

### 计算代码

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):
    vif_data = pd.DataFrame({
        'feature': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif_data.sort_values('VIF', ascending=False)

# 逐步剔除 VIF > 10 的特征（每次只删最高的，然后重新计算）
while calc_vif(X_woe_encoded)['VIF'].max() > 10:
    worst = calc_vif(X_woe_encoded).iloc[0]
    print(f"删除高共线性特征：{worst['feature']}（VIF={worst['VIF']:.1f}）")
    X_woe_encoded = X_woe_encoded.drop(columns=[worst['feature']])
```

---

## 九、指标使用时序：建模全流程对应

```
原始数据
    │
    ├─ 单变量分析 ──→ IV 筛选特征（IV < 0.02 丢弃，IV > 0.5 查泄露）
    │
    ├─ WOE 编码 ───→ 将特征转为 WOE 值，输入逻辑回归
    │
    ├─ 共线性检查 ──→ VIF（删除 VIF > 10 的特征）
    │
    ├─ 模型训练
    │
    ├─ 区分度评估 ──→ KS（风控行业标准）+ AUC（模型调参参考）
    │                  KS ≥ 0.3 才考虑上线
    │
    └─ 上线后监控 ──→ 每月 PSI（整体稳定性）+ CSI（特征漂移诊断）
                      PSI > 0.25 触发模型重评估
```
