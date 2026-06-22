---
title: "风控建模核心评价指标：WOE、IV、KS、AUC、PSI、Lift 全解"
category: modeling
summary: 系统介绍信贷风控建模中最常用的评价指标——WOE/IV（变量预测能力与筛选）、KS/AUC/Lift（模型区分度）、PSI（模型稳定性监控）——包含专业定义、公式推导、代码实现和判断阈值。
tags: [WOE, IV, KS, AUC, PSI, Lift, 模型评估, 风控建模, 评分卡]
---

## 一、指标体系概览

风控建模的评价指标体系覆盖建模的三个核心阶段：

| 阶段 | 指标 | 作用 |
|------|------|------|
| **变量筛选** | WOE、IV | 衡量各变量对因变量（y）的预测能力，筛选入模变量 |
| **模型区分度** | KS、AUC、Lift | 评估模型区分好坏客户的能力 |
| **模型稳定性** | PSI、CSI | 衡量模型评分分布是否发生偏移，监控线上模型的有效性 |

---

## 二、WOE：证据权重编码

### 定义与作用

WOE（Weight of Evidence，证据权重）是一种针对逻辑回归的特征编码方法，将原始变量各分箱的好坏客户分布转化为线性可用的数值，使逻辑回归能够捕捉变量与违约率之间的非线性关系。

```
WOE_i = ln( 坏客户占比_i / 好客户占比_i )
      = ln[ (B_i / B_total) / (G_i / G_total) ]

  B_i     ：第 i 个分箱内的坏客户数
  G_i     ：第 i 个分箱内的好客户数
  B_total ：全体坏客户数
  G_total ：全体好客户数
```

### 解读规则

| WOE 值 | 含义 |
|--------|------|
| WOE > 0 | 该分箱中坏客户集中度高于整体，违约风险偏高 |
| WOE = 0 | 该分箱好坏分布与整体一致，无区分信息 |
| WOE < 0 | 该分箱中好客户集中度高于整体，违约风险偏低 |
| 绝对值越大 | 该分箱的区分能力越强 |

WOE 值单调性也是评估分箱质量的标准之一：理想情况下，随着风险升高，WOE 应单调递增。

**示例**：历史逾期次数 WOE 编码

```
逾期 0 次 → WOE = -0.82（好客户集中）
逾期 1 次 → WOE = +0.31（略高风险）
逾期 2 次 → WOE = +0.97（高风险）
逾期 3 次以上 → WOE = +1.54（坏客户集中）
```

### 计算代码

```python
import numpy as np
import pandas as pd

def calc_woe_iv(df, feature, target, bins=10):
    """计算单个变量的 WOE 和 IV，返回分箱明细表和 IV 总值"""
    total_bad  = df[target].sum()
    total_good = (df[target] == 0).sum()

    df = df.copy()
    df['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')

    grouped = df.groupby('bin')[target].agg(['sum', 'count'])
    grouped.columns = ['bad', 'total']
    grouped['good']      = grouped['total'] - grouped['bad']
    grouped['bad_rate']  = grouped['bad']  / total_bad
    grouped['good_rate'] = grouped['good'] / total_good

    # 防止分箱内样本为 0 导致 ln(0)
    grouped['bad_rate']  = grouped['bad_rate'].replace(0, 0.0001)
    grouped['good_rate'] = grouped['good_rate'].replace(0, 0.0001)

    grouped['woe'] = np.log(grouped['bad_rate'] / grouped['good_rate'])
    grouped['iv']  = (grouped['bad_rate'] - grouped['good_rate']) * grouped['woe']

    return grouped[['bad', 'good', 'bad_rate', 'good_rate', 'woe', 'iv']], grouped['iv'].sum()
```

### WOE 编码的工程价值

1. **线性化**：将非线性的风险关系转化为逻辑回归可直接使用的线性输入
2. **缺失值处理**：缺失值独立成箱，计算其 WOE 值，无需额外填充
3. **异常值鲁棒**：分箱操作天然削弱极端值对模型的影响
4. **可解释性**：评分卡中每个变量的得分 = WOE × 系数，贡献清晰可见

---

## 三、IV：变量预测能力的量化指标

### 定义与作用

IV（Information Value，信息值）**衡量单个变量对因变量（y，即违约标签）的整体预测能力**，是变量筛选阶段的核心指标。在建模前，通常先计算所有候选变量的 IV，以此决定哪些变量值得进入后续建模流程。

```
IV = Σ (坏客户占比_i - 好客户占比_i) × WOE_i
   = Σ (B_i/B_total - G_i/G_total) × ln[ (B_i/B_total) / (G_i/G_total) ]
```

IV 是 WOE 的加权求和：某分箱的好坏客户差距越大（括号部分越大），WOE 绝对值越大，该分箱对 IV 的贡献就越多。

### 判断标准

| IV 值 | 变量预测能力 | 建议 |
|-------|-------------|------|
| < 0.02 | 无预测能力 | 丢弃 |
| 0.02 – 0.1 | 弱 | 谨慎使用 |
| 0.1 – 0.3 | 中等 | 可入模 |
| 0.3 – 0.5 | 强 | 优先入模 |
| > 0.5 | 异常强 | **必须排查数据泄露** |

> IV > 0.5 在干净的信贷数据中极为罕见。若出现，通常是变量与标签存在直接关联（如用"当月逾期天数"预测"是否逾期"），属于目标泄露，需立即剔除。

### 批量筛选变量

```python
iv_results = {}
for feature in candidate_features:
    try:
        _, iv = calc_woe_iv(df_train, feature, 'is_bad')
        iv_results[feature] = iv
    except Exception:
        iv_results[feature] = 0

iv_series = pd.Series(iv_results).sort_values(ascending=False)
print(iv_series.to_string())

# 保留 IV > 0.02 的变量
selected = iv_series[iv_series > 0.02].index.tolist()
```

---

## 四、KS：模型区分度的核心指标

### 定义与作用

KS（Kolmogorov-Smirnov 统计量）**用于评估模型区分好坏客户的能力**，是信贷风控建模中最常用的模型性能指标。KS 越高，模型将好坏客户"分得越开"，风控策略越容易在高精度和低误拒之间取得平衡。

```
KS = max | F_bad(x) - F_good(x) |

  F_bad(x)  ：坏客户评分的累积分布函数
  F_good(x) ：好客户评分的累积分布函数
  x 遍历所有评分阈值
```

### 直觉理解

将所有客户按预测违约概率从低到高排序，分别累积好客户和坏客户的占比，画出两条曲线。KS 即两条曲线之间的最大垂直距离：

```
累积占比
100% |                            _______ 坏客户曲线
     |                   _________
     |           ________
  0% |___________________________________________
     低风险评分              →              高风险评分

             好客户曲线（始终在坏客户曲线下方）

KS = 两条曲线最大垂直距离
     出现在该处意味着：以此评分为切点，好坏分离效果最佳
```

### 判断标准

| KS 值 | 评价 |
|-------|------|
| < 0.2 | 模型区分度差，不可用 |
| 0.2 – 0.3 | 一般 |
| 0.3 – 0.45 | 良好，达到上线标准 |
| 0.45 – 0.6 | 优秀 |
| > 0.6 | 极强，需排查过拟合或数据泄露 |

### 计算代码

```python
from scipy.stats import ks_2samp

def calc_ks(y_true, y_pred_proba):
    bad_scores  = y_pred_proba[y_true == 1]
    good_scores = y_pred_proba[y_true == 0]
    ks_stat, _ = ks_2samp(bad_scores, good_scores)
    return ks_stat

def ks_table(y_true, y_pred_proba, n_bins=10):
    """分档 KS 表，便于观察每档的好坏分布和最大 KS 出现位置"""
    df = pd.DataFrame({'score': y_pred_proba, 'label': y_true})
    df['bin'] = pd.qcut(df['score'], q=n_bins, duplicates='drop', labels=False)

    total_bad  = y_true.sum()
    total_good = (y_true == 0).sum()

    t = df.groupby('bin')['label'].agg(['sum', 'count'])
    t.columns = ['bad', 'total']
    t['good']          = t['total'] - t['bad']
    t['cum_bad_pct']   = t['bad'].cumsum()  / total_bad
    t['cum_good_pct']  = t['good'].cumsum() / total_good
    t['ks']            = abs(t['cum_bad_pct'] - t['cum_good_pct'])

    return t
```

---

## 五、AUC：模型区分度的全局度量

### 定义与作用

AUC（Area Under the ROC Curve）是 ROC 曲线下方的面积，从全局角度衡量模型对任意一对好坏客户的排序能力，不依赖于具体分箱，是模型调参和横向比较的标准指标。

```
AUC = P( 模型对随机一个坏客户的评分 > 对随机一个好客户的评分 )
```

即：随机抽取一名好客户和一名坏客户，模型正确判断坏客户风险更高的概率。

ROC 曲线的两个坐标轴定义：

```
TPR（True Positive Rate，召回率）= TP / (TP + FN)
    坏客户中被模型正确识别的比例

FPR（False Positive Rate，误拒率）= FP / (FP + TN)
    好客户中被模型错误标记为高风险的比例
```

### 判断标准

| AUC 值 | 评价 |
|--------|------|
| 0.5 | 无区分能力，等同随机猜测 |
| 0.6 – 0.7 | 一般 |
| 0.7 – 0.8 | 良好 |
| 0.8 – 0.9 | 优秀 |
| > 0.9 | 极强，警惕数据泄露 |

### KS 与 AUC 的关系

两者均衡量区分度，近似转换关系：

```
KS ≈ 2 × (AUC - 0.5)
```

| 指标 | 适用场景 |
|------|---------|
| KS | 风控报告与上线决策（业内通用，直觉强） |
| AUC | 模型调参与方案对比（数学性质更好，不受分箱影响） |

### 计算代码

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('FPR（误拒率）')
plt.ylabel('TPR（坏客户识别率）')
plt.legend()
plt.show()
```

> **Gini 系数（信贷语境）**：部分机构使用 Gini = 2 × AUC − 1，与决策树中的基尼不纯度是完全不同的概念，仅为 AUC 的线性变换，范围 [0, 1]。

---

## 六、Lift：模型对随机基准的提升倍数

### 定义与作用

Lift（提升度）**衡量模型相较于随机抽样对坏客户识别能力的提升倍数**，直接反映模型的业务价值。在实际风控中，Lift 曲线常用于回答"如果我只审核评分最高的前 X% 申请人，能捕获多少比例的坏客户"这类决策问题。

```
Lift@Top-X% = 模型前 X% 样本中坏客户占比 / 全量样本中坏客户占比
            = 捕获坏客户的精度 / 整体坏客户率
```

| Lift 值 | 含义 |
|---------|------|
| Lift = 1 | 模型与随机无差异 |
| Lift = 2 | 该分位的坏客户浓度是整体的 2 倍 |
| Lift 越大 | 模型在该分位的识别精度越高 |

### 直觉理解

假设整体坏客户率为 5%，模型评分最高的前 10% 申请人中，坏客户占比为 20%，则：

```
Lift@Top-10% = 20% / 5% = 4.0

含义：通过模型圈定的前 10% 高风险人群，
      坏客户浓度是随机抽取的 4 倍
```

这意味着，只审核这 10% 的高风险申请，就能用 1/10 的审核成本覆盖 4 倍于随机的坏客户拦截效果。

### Lift 曲线

Lift 曲线展示从高风险到低风险各分位段的提升倍数，通常横轴为"累积样本占比"，纵轴为对应的 Lift 值：

```
Lift
 4.0 |*
 3.5 | *
 3.0 |  *
 2.5 |   *
 2.0 |    **
 1.5 |      ***
 1.0 |─────────────── 随机基准线
 0.5 |
     |________________________
      10% 20% 30% ... 100%
      ← 高风险             低风险 →
```

### 计算代码

```python
def lift_table(y_true, y_pred_proba, n_bins=10):
    """生成 Lift 分析表"""
    df = pd.DataFrame({'prob': y_pred_proba, 'label': y_true})
    # 按评分从高到低排序
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)

    total        = len(df)
    total_bad    = y_true.sum()
    base_bad_rate = total_bad / total       # 整体坏客户率（随机基准）

    bin_size = total // n_bins
    records = []
    for i in range(n_bins):
        chunk     = df.iloc[i * bin_size: (i + 1) * bin_size]
        cum_chunk = df.iloc[: (i + 1) * bin_size]
        records.append({
            'decile'             : i + 1,
            'score_min'          : chunk['prob'].min(),
            'score_max'          : chunk['prob'].max(),
            'bad_in_bin'         : chunk['label'].sum(),
            'cum_bad'            : cum_chunk['label'].sum(),
            'cum_sample_pct'     : (i + 1) * bin_size / total,
            'cum_bad_capture_pct': cum_chunk['label'].sum() / total_bad,
            'lift'               : (cum_chunk['label'].mean()) / base_bad_rate,
        })

    return pd.DataFrame(records)

table = lift_table(y_test, y_pred_proba)
print(table[['decile', 'cum_sample_pct', 'cum_bad_capture_pct', 'lift']].to_string(index=False))
```

**输出示例**：

```
 decile  cum_sample_pct  cum_bad_capture_pct  lift
      1            0.10                 0.38   3.80
      2            0.20                 0.58   2.90
      3            0.30                 0.72   2.40
      4            0.40                 0.82   2.05
      ...
     10            1.00                 1.00   1.00
```

### Lift 在风控决策中的应用

| 应用场景 | 说明 |
|----------|------|
| 确定拒绝策略 | 选取 Lift > 2 的评分段作为高风险拒绝区间 |
| 量化模型价值 | 向业务团队展示"审核前 20% 能捕获 58% 坏客户" |
| 多模型对比 | 同等覆盖率下，Lift 越高的模型精度越好 |

---

## 七、PSI：模型稳定性的核心监控指标

### 定义与作用

PSI（Population Stability Index，群体稳定性指数）**用于衡量模型评分分布的稳定性**，即模型上线后，线上评分分布是否相对建模期发生了显著偏移。PSI 是判断模型是否仍然有效、是否需要重训的标准依据。

```
PSI = Σ (A_i - E_i) × ln(A_i / E_i)

  E_i ：建模期（训练/验证集）第 i 分段的客户占比（预期分布）
  A_i ：线上某监控周期第 i 分段的客户占比（实际分布）
```

PSI 度量两个分布之间的对称 KL 散度，E_i 与 A_i 差距越大，PSI 越高。

### 判断标准

| PSI 值 | 稳定性 | 操作建议 |
|--------|--------|---------|
| < 0.1 | 稳定 | 正常，无需动作 |
| 0.1 – 0.25 | 轻微偏移 | 关注并排查原因 |
| > 0.25 | 显著偏移 | 模型可能失效，评估是否重训 |

### 计算代码

```python
def calc_psi(expected, actual, n_bins=10):
    """
    expected: 建模期评分数组
    actual:   监控期评分数组
    """
    # 以建模期分位数定义统一的分箱边界
    breakpoints    = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    def bin_rates(scores):
        counts = np.histogram(scores, bins=breakpoints)[0]
        rates  = counts / len(scores)
        return np.where(rates == 0, 0.0001, rates)

    e = bin_rates(expected)
    a = bin_rates(actual)

    return float(np.sum((a - e) * np.log(a / e)))

# 月度监控示例
for month, group in df_prod.groupby('month'):
    psi = calc_psi(train_scores, group['score'].values)
    status = 'ALERT' if psi > 0.25 else ('WARN' if psi > 0.1 else 'OK')
    print(f"{month}  PSI={psi:.4f}  [{status}]")
```

### PSI 偏移的常见根因

| 根因类型 | 具体表现 |
|----------|---------|
| 客群结构变化 | 拓展新渠道或新产品，申请人特征与建模期不同 |
| 宏观经济冲击 | 经济周期下行导致整体信用水平系统性下移 |
| 特征数据质量 | 上游数据源断供或字段口径变更 |
| 季节性效应 | 特定时段（如年末）申请人结构周期性变化 |

---

## 八、CSI：特征稳定性诊断工具

CSI（Characteristic Stability Index）与 PSI 公式完全相同，区别在于监控对象是**单个入模变量的分布**而非整体评分分布：

```
CSI_j = Σ (A_i - E_i) × ln(A_i / E_i)   （针对变量 j 的各分箱）
```

**用途**：当 PSI 超标时，逐一计算所有入模变量的 CSI，CSI 最高的变量即为最可能的漂移根因。

```python
csi_results = {
    feat: calc_psi(df_train[feat].values, df_prod_latest[feat].values)
    for feat in model_features
}
csi_series = pd.Series(csi_results).sort_values(ascending=False)
print("CSI Top 10（PSI 超标根因排查）：")
print(csi_series.head(10).to_string())
```

---

## 九、VIF：入模变量共线性检验

VIF（Variance Inflation Factor，方差膨胀因子）检测逻辑回归入模变量之间是否存在多重共线性。共线性会导致系数符号反转或数值极度不稳定，使模型失去可解释性。

```
VIF_j = 1 / (1 - R²_j)

  R²_j：以变量 j 为因变量、其余入模变量为自变量进行回归的决定系数
```

| VIF 值 | 含义 |
|--------|------|
| 1 | 与其他变量无相关 |
| 1 – 5 | 轻微相关，可接受 |
| 5 – 10 | 中度共线，需关注 |
| > 10 | 严重共线，需剔除 |

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):
    return pd.DataFrame({
        'feature': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }).sort_values('VIF', ascending=False).reset_index(drop=True)

# 逐步剔除：每轮只删 VIF 最高的变量，重新计算，直至所有变量 VIF < 10
while True:
    vif_df = calc_vif(X_woe)
    if vif_df['VIF'].max() <= 10:
        break
    drop_feat = vif_df.iloc[0]['feature']
    print(f"剔除：{drop_feat}（VIF = {vif_df.iloc[0]['VIF']:.2f}）")
    X_woe = X_woe.drop(columns=[drop_feat])
```

---

## 十、建模全流程中的指标使用时序

```
候选变量池
    │
    ├─ 1. 单变量 IV 筛选
    │      IV < 0.02  → 丢弃（无预测能力）
    │      IV > 0.5   → 排查目标泄露
    │
    ├─ 2. WOE 编码
    │      将各变量转为 WOE 值，确认单调性，输入逻辑回归
    │
    ├─ 3. VIF 共线性检验
    │      VIF > 10 的变量逐步剔除
    │
    ├─ 4. 模型训练
    │
    ├─ 5. 区分度评估（样本外测试集）
    │      KS ≥ 0.3   → 满足上线基准
    │      AUC        → 辅助调参与模型对比
    │      Lift 曲线  → 确定最优风险切点，量化业务价值
    │
    └─ 6. 上线后稳定性监控（每月）
           PSI > 0.1  → 预警，排查根因
           PSI > 0.25 → 模型失效，启动重训流程
           CSI        → 定位发生漂移的具体变量
```
