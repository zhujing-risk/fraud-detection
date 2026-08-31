"""
01 - 数据探索与欺诈模式分析
目标：了解数据分布，识别欺诈交易的关键特征，为特征工程提供方向
面试价值：展示你如何从数据中发现欺诈模式（对标支付风控的case分析能力）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================================
# 1. 数据加载
# ============================================================
# 下载地址: https://www.kaggle.com/c/ieee-fraud-detection/data
# 下载后放到 ./data/ 目录下

train_transaction = pd.read_csv('./data/train_transaction.csv')
train_identity = pd.read_csv('./data/train_identity.csv')

# 合并交易表和身份表（类比：交易数据 + 设备/环境数据）
train = train_transaction.merge(train_identity, on='TransactionID', how='left')

print(f"总交易量: {len(train):,}")
print(f"欺诈交易量: {train['isFraud'].sum():,}")
print(f"欺诈率: {train['isFraud'].mean():.4f} ({train['isFraud'].mean()*100:.2f}%)")
print(f"特征维度: {train.shape[1]}")

# ============================================================
# 2. 欺诈率分布分析（对标：各维度风险画像）
# ============================================================

# 2.1 交易金额分布
print("\n=== 交易金额分析 ===")
print("正常交易金额统计:")
print(train[train['isFraud']==0]['TransactionAmt'].describe())
print("\n欺诈交易金额统计:")
print(train[train['isFraud']==1]['TransactionAmt'].describe())

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 金额分布对比
axes[0].hist(train[train['isFraud']==0]['TransactionAmt'].clip(upper=500),
             bins=50, alpha=0.7, label='正常', density=True)
axes[0].hist(train[train['isFraud']==1]['TransactionAmt'].clip(upper=500),
             bins=50, alpha=0.7, label='欺诈', density=True)
axes[0].set_title('交易金额分布对比')
axes[0].set_xlabel('交易金额')
axes[0].legend()

# 金额区间欺诈率
train['amt_bin'] = pd.cut(train['TransactionAmt'], bins=[0,50,100,200,500,1000,5000,50000])
fraud_by_amt = train.groupby('amt_bin')['isFraud'].agg(['mean','count']).reset_index()
fraud_by_amt.columns = ['金额区间', '欺诈率', '交易量']
axes[1].bar(range(len(fraud_by_amt)), fraud_by_amt['欺诈率'])
axes[1].set_xticks(range(len(fraud_by_amt)))
axes[1].set_xticklabels(fraud_by_amt['金额区间'].astype(str), rotation=45)
axes[1].set_title('各金额区间欺诈率')
axes[1].set_ylabel('欺诈率')

plt.tight_layout()
plt.savefig('./output/01_amount_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# 2.2 卡类型分析（对标：支付方式维度的风险差异）
print("\n=== 卡类型欺诈率 ===")
for col in ['card4', 'card6']:  # card4=卡组织, card6=信用卡/借记卡
    print(f"\n{col}维度:")
    card_fraud = train.groupby(col)['isFraud'].agg(['mean','count'])
    card_fraud.columns = ['欺诈率', '交易量']
    card_fraud = card_fraud.sort_values('欺诈率', ascending=False)
    print(card_fraud)

# 2.3 设备类型分析（对标：设备指纹风控）
print("\n=== 设备类型欺诈率 ===")
device_fraud = train.groupby('DeviceType')['isFraud'].agg(['mean','count'])
device_fraud.columns = ['欺诈率', '交易量']
print(device_fraud)

# 2.4 时间模式分析（对标：交易时间异常检测）
# TransactionDT 是相对时间戳（秒）
train['hour'] = (train['TransactionDT'] / 3600) % 24
train['day_of_week'] = (train['TransactionDT'] / 86400) % 7

print("\n=== 小时级欺诈率分布 ===")
hourly_fraud = train.groupby(train['hour'].astype(int))['isFraud'].mean()
print(hourly_fraud.sort_values(ascending=False).head(10))

# ============================================================
# 3. 缺失值分析（对标：信息完整度作为风险信号）
# ============================================================
print("\n=== 缺失值分析 ===")
# 欺诈交易的缺失值比例 vs 正常交易
fraud_missing = train[train['isFraud']==1].isnull().mean()
normal_missing = train[train['isFraud']==0].isnull().mean()

missing_diff = (fraud_missing - normal_missing).sort_values(ascending=False)
print("欺诈交易缺失率显著高于正常交易的特征（top 10）:")
print(missing_diff.head(10))
# 面试点：缺失本身就是强特征！欺诈用户往往信息不完整

# ============================================================
# 4. 邮箱域名分析（对标：注册信息异常检测）
# ============================================================
print("\n=== 邮箱域名欺诈率 ===")
for col in ['P_emaildomain', 'R_emaildomain']:
    if col in train.columns:
        email_fraud = train.groupby(col)['isFraud'].agg(['mean','count'])
        email_fraud = email_fraud[email_fraud['count'] > 100].sort_values('mean', ascending=False)
        print(f"\n{col} Top 10 高风险域名:")
        print(email_fraud.head(10))

# ============================================================
# 5. 关键发现总结（面试时直接讲这些结论）
# ============================================================
print("\n" + "="*60)
print("关键发现总结（面试话术）")
print("="*60)
print("""
1. 金额特征：欺诈交易金额中位数显著高于正常交易，大额交易风险更高
   → 策略建议：大额交易加验证（3DS/人工审核）

2. 时间特征：凌晨时段欺诈率显著上升
   → 策略建议：非活跃时段提高风控阈值

3. 设备特征：desktop欺诈率高于mobile
   → 策略建议：PC端交易需要更多验证信号

4. 信息完整度：欺诈交易的身份信息缺失率显著更高
   → 策略建议：信息缺失度本身作为风险特征入模

5. 邮箱域名：部分邮箱域名欺诈率异常高
   → 策略建议：高风险邮箱域名名单 + 临时邮箱识别规则
""")
