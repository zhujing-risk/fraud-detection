"""
02 - 特征工程
目标：构建支付风控场景下的核心特征，展示你对"什么信号能识别欺诈"的理解
面试价值：展示特征工程的业务直觉 + 技术落地能力

特征分类（对标真实支付风控系统）：
1. 交易特征：金额、频率、时间
2. 用户行为特征：历史统计、偏离度
3. 设备/环境特征：设备指纹、IP、地理位置
4. 关联特征：卡号-设备-地址的关联网络
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 数据加载
# ============================================================
train_transaction = pd.read_csv('./data/train_transaction.csv')
train_identity = pd.read_csv('./data/train_identity.csv')
train = train_transaction.merge(train_identity, on='TransactionID', how='left')

# ============================================================
# 1. 交易金额特征（对标：交易金额异常检测）
# ============================================================
print("构建交易金额特征...")

# 金额本身
train['amt_log'] = np.log1p(train['TransactionAmt'])

# 金额是否为整数（欺诈交易常为整数金额）
train['amt_is_round'] = (train['TransactionAmt'] % 1 == 0).astype(int)

# 金额小数部分（某些欺诈有特定小数模式）
train['amt_decimal'] = train['TransactionAmt'] - train['TransactionAmt'].astype(int)

# ============================================================
# 2. 时间窗口聚合特征（对标：实时特征计算 - 滑动窗口）
# ============================================================
print("构建时间窗口特征...")

# 将相对时间转为小时/天
train['hour'] = (train['TransactionDT'] / 3600) % 24
train['day'] = train['TransactionDT'] // 86400
train['day_of_week'] = train['day'] % 7

# 是否凌晨交易（高风险时段）
train['is_night'] = ((train['hour'] >= 0) & (train['hour'] <= 5)).astype(int)

# 是否周末
train['is_weekend'] = (train['day_of_week'] >= 5).astype(int)

# 同一张卡的交易频率（时间窗口聚合 - 模拟实时特征）
# 面试点：真实系统中这些是实时计算的（如最近1h/6h/24h的交易次数/金额）
for col in ['card1', 'card2']:
    # 同卡交易次数
    card_counts = train.groupby(col)['TransactionID'].transform('count')
    train[f'{col}_count'] = card_counts

    # 同卡平均金额
    card_mean_amt = train.groupby(col)['TransactionAmt'].transform('mean')
    train[f'{col}_mean_amt'] = card_mean_amt

    # 当前金额 vs 该卡历史平均金额的偏离度（核心特征！）
    # 面试点：金额偏离度是支付风控最重要的特征之一
    train[f'{col}_amt_deviation'] = (
        train['TransactionAmt'] - card_mean_amt
    ) / (card_mean_amt + 1)

    # 同卡最大金额
    card_max_amt = train.groupby(col)['TransactionAmt'].transform('max')
    train[f'{col}_amt_ratio_to_max'] = train['TransactionAmt'] / (card_max_amt + 1)

# ============================================================
# 3. 设备/环境特征（对标：设备指纹风控）
# ============================================================
print("构建设备环境特征...")

# 设备类型编码
train['DeviceType_encoded'] = LabelEncoder().fit_transform(
    train['DeviceType'].fillna('unknown').astype(str)
)

# 设备信息缺失度（关键信号：欺诈者往往隐藏设备信息）
identity_cols = [c for c in train.columns if c.startswith('id_')]
train['identity_missing_count'] = train[identity_cols].isnull().sum(axis=1)
train['identity_missing_ratio'] = train['identity_missing_count'] / len(identity_cols)

# 浏览器信息缺失
train['device_info_missing'] = train['DeviceInfo'].isnull().astype(int)

# ============================================================
# 4. 地址匹配特征（对标：收货地址与注册地址一致性）
# ============================================================
print("构建地址匹配特征...")

# 账单地址 vs 收货地址是否匹配
train['addr_match'] = (train['addr1'] == train['addr2']).astype(int)

# 地址缺失
train['addr1_missing'] = train['addr1'].isnull().astype(int)
train['addr2_missing'] = train['addr2'].isnull().astype(int)

# ============================================================
# 5. 邮箱特征（对标：注册信息风险评估）
# ============================================================
print("构建邮箱特征...")

# 邮箱域名提取
train['P_email_domain_suffix'] = train['P_emaildomain'].apply(
    lambda x: str(x).split('.')[-1] if pd.notna(x) else 'missing'
)

# 是否使用免费邮箱（gmail/yahoo/hotmail等 vs 企业邮箱）
free_emails = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
train['is_free_email'] = train['P_emaildomain'].isin(free_emails).astype(int)

# 发送方邮箱 vs 接收方邮箱是否一致
train['email_match'] = (train['P_emaildomain'] == train['R_emaildomain']).astype(int)

# ============================================================
# 6. 关联网络特征（对标：图关联分析识别团伙欺诈）
# ============================================================
print("构建关联网络特征...")

# 同一邮箱关联多少张卡（团伙特征：一个邮箱绑多卡）
email_card_count = train.groupby('P_emaildomain')['card1'].transform('nunique')
train['email_card_nunique'] = email_card_count

# 同一设备关联多少张卡
if 'DeviceInfo' in train.columns:
    device_card_count = train.groupby('DeviceInfo')['card1'].transform('nunique')
    train['device_card_nunique'] = device_card_count

# 同一张卡关联多少个地址
card_addr_count = train.groupby('card1')['addr1'].transform('nunique')
train['card_addr_nunique'] = card_addr_count

# ============================================================
# 7. V系列特征降维（匿名特征处理）
# ============================================================
print("处理V系列匿名特征...")

v_cols = [c for c in train.columns if c.startswith('V')]
# 计算与target的相关性，保留高相关特征
if len(v_cols) > 0:
    v_corr = train[v_cols + ['isFraud']].corr()['isFraud'].drop('isFraud').abs()
    top_v_features = v_corr.sort_values(ascending=False).head(50).index.tolist()
    print(f"保留Top 50相关V特征（共{len(v_cols)}个）")

# ============================================================
# 8. 特征汇总
# ============================================================
new_features = [
    # 金额特征
    'amt_log', 'amt_is_round', 'amt_decimal',
    # 时间特征
    'hour', 'day_of_week', 'is_night', 'is_weekend',
    # 卡维度聚合
    'card1_count', 'card1_mean_amt', 'card1_amt_deviation', 'card1_amt_ratio_to_max',
    'card2_count', 'card2_mean_amt', 'card2_amt_deviation', 'card2_amt_ratio_to_max',
    # 设备/环境
    'DeviceType_encoded', 'identity_missing_count', 'identity_missing_ratio',
    'device_info_missing',
    # 地址
    'addr_match', 'addr1_missing', 'addr2_missing',
    # 邮箱
    'is_free_email', 'email_match',
    # 关联网络
    'email_card_nunique', 'card_addr_nunique',
]

print(f"\n新增特征数: {len(new_features)}")
print(f"总特征数（含原始+新增）: {train.shape[1]}")

# 保存
train.to_pickle('./data/train_featured.pkl')
print("特征工程完成，已保存到 ./data/train_featured.pkl")

# ============================================================
# 面试讲述要点
# ============================================================
print("""
\n=== 面试讲述框架 ===

特征工程思路（按风控业务逻辑组织，不是按技术分类）：

1. 交易本身是否异常？
   → 金额偏离历史均值、是否整数金额、非活跃时段

2. 用户身份是否可信？
   → 设备信息完整度、邮箱类型、地址一致性

3. 是否存在团伙特征？
   → 同设备多卡、同邮箱多卡、同卡多地址

4. 行为模式是否偏离？
   → 当前金额vs历史均值偏离度、交易频率突增

关键点：特征工程要用业务逻辑驱动，而不是盲目暴力衍生。
面试时先讲"我要检测什么"，再讲"用什么特征检测"。
""")
