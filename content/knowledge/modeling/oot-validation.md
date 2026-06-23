---
order: 2
title: "评分卡样本构建与跨时间验证（OOT）"
category: modeling
summary: 系统介绍信贷评分卡开发中的样本构建方法——观察窗口与表现窗口的定义、好坏灰三类账户划分、训练/验证/OOT 切分方案，以及如何通过 OOT 评估模型的跨时间泛化能力，识别数据泄露与模型衰退。
tags: [OOT, 跨时间验证, 观察窗口, 表现窗口, 评分卡, 样本构建, Vintage分析]
---

## 一、为什么不能随机切分样本

在通用机器学习场景中，随机将数据集 8:2 分成训练集和测试集是标准做法。但在信贷风控建模中，这样做会产生严重问题：

**信贷数据具有强时序性**：
- 今天申请的客户，其违约行为可能发生在 3 个月、6 个月甚至 12 个月后
- 不同时期的申请人群体结构可能因渠道变化、经济周期、政策调整而显著不同
- 模型需要的是"对未来申请人的预测能力"，而非"对历史数据的记忆能力"

随机切分的问题：
```
随机切分：
  训练集 ←── 随机混合 ──→ 测试集
  2022年1月的申请人 与 2023年6月的申请人 混在一起
  → 测试集 KS 虚高，上线后真实表现远低于测试结果
  → 本质上是"用未来信息预测过去"，是一种隐性的时序泄露

按时间切分（正确做法）：
  训练集（历史）→ 验证集（稍新）→ OOT（最新）
  → 真实模拟"用历史数据建模，预测未来申请人"的生产场景
```

---

## 二、观察窗口与表现窗口

评分卡样本构建的核心是定义两个时间窗口：

### 观察窗口（Observation Window）

**定义**：采集申请人特征数据的时间点，即"观察时刻"。

```
观察时刻 = 申请时间点（或账户开立时间点）

在该时刻，我们能观察到：
  ✓ 申请人的基本信息（年龄、收入、职业）
  ✓ 历史信用行为（过去 12 个月的逾期记录）
  ✓ 负债情况（当前负债、申请金额）
  ✗ 不能用观察时刻之后的任何信息（这是数据泄露）
```

### 表现窗口（Performance Window）

**定义**：观察申请人是否违约的时间段，即"好坏账户"判断的依据窗口。

```
表现窗口 = 观察时刻 + N 个月

常见设置：
  消费金融短期贷款（3–12 期）：表现窗口 = 6 个月
  信用卡 / 中长期贷款：表现窗口 = 12 个月
  住房贷款：表现窗口 = 24 个月

判断标准（以 6 个月为例）：
  观察时刻后 6 个月内，出现 M2+（逾期 60 天以上）→ 坏账（Bad）
  观察时刻后 6 个月内，始终正常还款 → 好账（Good）
```

### 时间轴示意

```
时间轴 ──────────────────────────────────────────────────────→

  申请时刻           +6个月             +12个月
      │               │                  │
      ▼               ▼                  ▼
  ┌───────┐       ┌────────────────────┐
  │ 观察   │       │    表现窗口（6M）   │
  │ 窗口   │       │  在此期间是否逾期？ │
  └───────┘       └────────────────────┘

  特征提取时间点 ──→ 若 6M 内 M2+ = 坏账（label=1）
                     若 6M 内无逾期 = 好账（label=0）
                     若 6M 内 M1（轻微逾期）= 灰色账户（排除）
```

---

## 三、好坏灰三类账户的划分

### 标准定义

| 账户类型 | 判断标准 | 建模处理 |
|----------|---------|---------|
| **坏账（Bad）** | 表现窗口内达到 M2+（逾期 ≥ 60 天）或 M3+ | 标签 y = 1 |
| **好账（Good）** | 表现窗口内无逾期或仅有 M0（未逾期） | 标签 y = 0 |
| **灰色账户（Indeterminate）** | 表现窗口内出现 M1（逾期 30–59 天）但未达 M2+ | **剔除，不参与建模** |

### 为什么要剔除灰色账户

灰色账户的风险特征模糊：
- 行为上介于好坏之间，历史上约 30%–50% 的 M1 客户后续会转为 M2+
- 强行将其归入好账会低估真实坏率，归入坏账会高估
- 保留灰色账户会引入噪声，降低模型的区分精度

**坏账定义的选择影响**：

| 坏账定义 | 坏率水平 | 模型特点 |
|----------|---------|---------|
| M1+（30 天以上） | 坏率高（8%–20%） | 早期风险信号，模型更敏感，但标签噪声大 |
| M2+（60 天以上） | 中等（3%–10%） | 业界最常用，平衡了稳定性与敏感性 |
| M3+（90 天以上） | 低（1%–5%） | 标签最稳定，但错过早期风险，表现窗口需更长 |

### 提前终止账户的处理

表现窗口内提前还清贷款的账户，不能判断其是否"本来会违约"：

```
处理方式：
  ① 若提前还款时间 > 表现窗口 50%：按好账处理（有足够表现时长）
  ② 若提前还款时间 < 表现窗口 50%：剔除（表现时间不足，无法判断）
```

---

## 四、样本切分方案

### 标准三段式切分

```
时间轴 ──────────────────────────────────────────────────────→

  2022-01    2023-06    2023-09    2023-12
      │          │          │          │
      ▼          ▼          ▼          ▼
  ┌────────────────┐ ┌──────┐ ┌──────────┐
  │   训练集        │ │验证集│ │  OOT     │
  │  (In-Time)     │ │(Dev) │ │(Out-of-  │
  │  18 个月        │ │3 个月│ │  Time)   │
  │                │ │      │ │ 3 个月    │
  └────────────────┘ └──────┘ └──────────┘
        ↑                          ↑
   用于模型训练               用于评估模型
   参数学习                   跨时间泛化能力
```

**各数据集的作用**：

| 数据集 | 时间范围 | 作用 |
|--------|---------|------|
| 训练集（Train） | 最早的 12–18 个月 | 模型参数学习 |
| 验证集（Validation） | 训练集之后 3–6 个月 | Early Stopping、调参 |
| OOT（Out-of-Time） | 验证集之后的最新 3–6 个月 | 评估模型对"未来"的泛化能力 |

**注意**：验证集必须时间上晚于训练集，OOT 必须时间上晚于验证集——这是时序隔离的基本要求。

### 代码实现

```python
import pandas as pd

# 按申请月份切分
df['apply_month'] = pd.to_datetime(df['apply_date']).dt.to_period('M')

train_end = '2023-06'
val_end   = '2023-09'

df_train = df[df['apply_month'] <= train_end]
df_val   = df[(df['apply_month'] > train_end) & (df['apply_month'] <= val_end)]
df_oot   = df[df['apply_month'] > val_end]

print(f"训练集：{len(df_train):,} 条，坏率 {df_train['is_bad'].mean():.2%}")
print(f"验证集：{len(df_val):,} 条，坏率 {df_val['is_bad'].mean():.2%}")
print(f"OOT ：{len(df_oot):,} 条，坏率 {df_oot['is_bad'].mean():.2%}")
```

---

## 五、OOT 评估：检验模型的跨时间泛化能力

### 核心评估指标

```python
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import numpy as np

def evaluate_model(model, X, y, label=''):
    proba = model.predict_proba(X)[:, 1]
    auc   = roc_auc_score(y, proba)
    ks, _ = ks_2samp(proba[y == 1], proba[y == 0])
    print(f"{label:8s} | 样本量={len(y):,} | 坏率={y.mean():.2%} | AUC={auc:.4f} | KS={ks:.4f}")
    return proba

train_proba = evaluate_model(model, X_train, y_train, '训练集')
val_proba   = evaluate_model(model, X_val,   y_val,   '验证集')
oot_proba   = evaluate_model(model, X_oot,   y_oot,   'OOT')
```

**输出示例**：
```
训练集   | 样本量=45,231 | 坏率=5.82% | AUC=0.8634 | KS=0.5312
验证集   | 样本量=8,104  | 坏率=6.11% | AUC=0.8421 | KS=0.5087
OOT      | 样本量=7,893  | 坏率=6.43% | AUC=0.8198 | KS=0.4831
```

### KS 衰减判断标准

OOT KS 相对训练集 KS 下降是正常现象（因为模型没有"见过" OOT 数据），关键是下降幅度：

```
KS 衰减率 = (训练集 KS - OOT KS) / 训练集 KS
```

| KS 衰减率 | 评价 | 操作建议 |
|-----------|------|---------|
| < 5% | 模型极稳定 | 正常上线 |
| 5% – 15% | 正常衰减 | 可上线，关注后续 PSI |
| 15% – 25% | 衰减偏大 | 检查特征稳定性，谨慎上线 |
| > 25% | 衰减严重 | 排查过拟合或特征泄露，不建议上线 |

### OOT PSI：评估评分分布偏移

```python
def calc_psi(expected, actual, n_bins=10):
    breakpoints    = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    def bin_rates(scores):
        counts = np.histogram(scores, bins=breakpoints)[0]
        rates  = counts / len(scores)
        return np.where(rates == 0, 0.0001, rates)

    e, a = bin_rates(expected), bin_rates(actual)
    return float(np.sum((a - e) * np.log(a / e)))

# 用训练集评分作为基准，计算 OOT 的 PSI
psi_oot = calc_psi(train_proba, oot_proba)
print(f"OOT PSI: {psi_oot:.4f}")
# PSI < 0.1：分布稳定，模型在 OOT 上依然有效
# PSI > 0.25：分布偏移显著，需排查客群变化
```

---

## 六、Vintage 分析：追踪各期坏账的成熟曲线

Vintage 分析（账龄分析）是评分卡开发中另一个重要工具，用于观察**不同时期（批次）放款的坏账率随账龄的演变规律**，回答"这批贷款发放后，坏账是如何随时间累积的？"

### 核心概念

```
Vintage   = 某一时间段内放款的贷款批次（如 2023Q1 放款）
账龄（MOB）= Month on Book，贷款发放后经过的月数
累积坏账率  = 截至当前账龄，该 Vintage 中已发生 M2+ 的贷款占比
```

### Vintage 曲线的意义

```
累积坏账率
  6% |           2022Q3 ──────────────────•
     |       2022Q4 ──────────────────•
  4% |   2023Q1 ──────────────────•
     | 2023Q2 ──────────────•
  2% |
     |___________________________
      0  3  6  9  12  15  18（月/MOB）

若各期 Vintage 曲线基本平行：说明不同时期放款质量稳定
若某期 Vintage 曲线明显高于其他：该期存在信用异常（渠道变化？政策松动？）
若最新 Vintage 曲线斜率明显陡峭：预警宏观经济恶化或客群质量下降
```

### 计算代码

```python
def vintage_analysis(df, loan_date_col, event_date_col, target_col):
    """
    df             ：贷款数据
    loan_date_col  ：放款日期列
    event_date_col ：逾期发生日期列（无逾期则为 NaT）
    target_col     ：是否坏账（1/0）
    """
    df = df.copy()
    df['vintage']  = pd.to_datetime(df[loan_date_col]).dt.to_period('Q')
    df['mob']      = ((pd.to_datetime(df[event_date_col]).fillna(pd.Timestamp.now())
                       - pd.to_datetime(df[loan_date_col]))
                      .dt.days // 30).clip(lower=0)

    results = []
    for vintage, group in df.groupby('vintage'):
        total = len(group)
        for mob in range(1, 19):
            bad_by_mob = group[group['mob'] <= mob][target_col].sum()
            results.append({
                'vintage':        str(vintage),
                'mob':            mob,
                'cum_bad_rate':   bad_by_mob / total
            })

    return pd.DataFrame(results).pivot(index='mob', columns='vintage', values='cum_bad_rate')

vintage_table = vintage_analysis(df, 'loan_date', 'overdue_date', 'is_bad')

import matplotlib.pyplot as plt
vintage_table.plot(figsize=(12, 6), marker='o', markersize=3)
plt.xlabel('Month on Book (MOB)')
plt.ylabel('累积坏账率')
plt.title('Vintage 分析——各期放款累积坏账率')
plt.legend(title='放款批次', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
```

---

## 七、OOT 验证的常见坑

### 1. 表现窗口末端的"未成熟账户"问题

最新一批申请人的贷款可能还没到观察期结束，坏账还没完全暴露：

```
问题场景：
  表现窗口 = 12 个月
  最新申请时间 = 2023-12
  数据截止时间 = 2024-06
  → 该批账户只有 6 个月的表现记录，坏账尚未成熟
  → 坏率被低估，OOT 性能虚高

解决方案：
  OOT 样本的申请时间 + 表现窗口 ≤ 数据截止时间
  即：OOT 应取申请时间距今已超过完整表现窗口的账户
```

### 2. OOT 与训练集坏率差异过大

坏率相差过大时，KS/AUC 的直接对比会产生误导：

```python
# 检查各数据集的坏率
for name, df_set in [('训练集', df_train), ('验证集', df_val), ('OOT', df_oot)]:
    bad_rate = df_set['is_bad'].mean()
    print(f"{name}: 坏率 = {bad_rate:.2%}")

# 若 OOT 坏率与训练集相差 > 50%（相对值），需重新审视 OOT 的定义和覆盖时间
```

### 3. 特征在 OOT 期间口径变化

上游数据源在 OOT 期间发生了字段逻辑变更，导致特征分布突变：

```python
# OOT 前必须检查各入模特征的分布稳定性（CSI）
from evaluation_utils import calc_psi

for feat in model_features:
    csi = calc_psi(df_train[feat].values, df_oot[feat].values)
    if csi > 0.1:
        print(f"[警告] {feat}: CSI={csi:.4f}，特征分布发生偏移")
```

### 4. 混用观察时点特征

在构建训练集时，误用了"观察时刻之后才能知道"的特征：

```
典型案例：
  ✗ 用"该贷款最终是否提前还款"预测违约（未来信息）
  ✗ 用"当期实际还款金额"预测违约（还款行为本身就是结果）
  ✓ 只使用申请时刻及其之前的历史特征
```

### 5. OOT 样本量不足

OOT 样本量太少时，KS 估计不稳定：

```
建议 OOT 中坏样本数量 ≥ 200
若信贷坏率为 5%，则 OOT 样本总量建议 ≥ 4,000 条
```

---

## 八、OOT 在完整建模流程中的位置

```
数据准备
    │
    ├─ 定义好坏灰账户（M2+ = 坏，M1 = 灰，剔除灰色账户）
    │
    ├─ 确认表现窗口（确保 OOT 账户已完全成熟）
    │
    ├─ 按时间切分：训练集 / 验证集 / OOT
    │
    ├─ 特征工程（WOE 编码等，仅在训练集上 fit，应用到所有集）
    │
    ├─ 模型训练（训练集）+ 调参（验证集）
    │
    ├─ In-Time 评估：KS(训练) / KS(验证)
    │
    ├─ OOT 评估
    │     KS 衰减率 < 15%？ → 通过
    │     OOT PSI < 0.1？   → 通过
    │     各特征 CSI 正常？  → 通过
    │
    ├─ Vintage 分析：各期放款坏账曲线是否平行？
    │
    └─ 上线决策：OOT 各项指标均通过 → 上线
                 部分指标异常 → 排查 → 修正 → 重新 OOT 评估
```
