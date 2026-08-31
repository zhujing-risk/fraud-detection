"""
03 - 模型训练与评估
目标：训练欺诈检测模型，重点展示模型评估思路（不同于信贷，支付风控更关注Precision-Recall）
面试价值：展示你理解支付风控场景下模型评估的特殊性
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, classification_report,
    confusion_matrix, f1_score, average_precision_score
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 数据准备
# ============================================================
print("加载特征工程后的数据...")
train = pd.read_pickle('./data/train_featured.pkl')

# 选择特征（排除ID和target）
exclude_cols = ['TransactionID', 'TransactionDT', 'isFraud']
cat_cols = ['card4', 'card6', 'P_emaildomain', 'R_emaildomain',
            'DeviceType', 'DeviceInfo', 'ProductCD']

# 数值特征
num_features = [c for c in train.columns
                if c not in exclude_cols
                and train[c].dtype in ['float64', 'int64', 'float32', 'int32']
                and c not in cat_cols]

# 类别特征编码
for col in cat_cols:
    if col in train.columns:
        train[col] = train[col].astype('category')

features = num_features + [c for c in cat_cols if c in train.columns]
print(f"使用特征数: {len(features)}")

X = train[features]
y = train['isFraud']

print(f"正样本比例: {y.mean():.4f}")
print(f"正负样本比: 1:{int((1-y.mean())/y.mean())}")

# ============================================================
# 2. LightGBM模型训练（5折交叉验证）
# ============================================================
print("\n开始训练 LightGBM...")

# 面试点：参数设计的业务考量
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.01,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'min_child_samples': 100,
    'n_estimators': 2000,
    'early_stopping_rounds': 100,
    # 处理样本不均衡：正样本权重放大
    # 面试点：为什么不用过采样？因为支付风控数据量大，加权比SMOTE效率更高
    'scale_pos_weight': (1 - y.mean()) / y.mean(),
    'verbose': -1,
}

# 5折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
feature_importance = pd.DataFrame()

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1}/5 ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(200)]
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    # 特征重要度
    fold_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_,
        'fold': fold + 1
    })
    feature_importance = pd.concat([feature_importance, fold_importance])

    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

# ============================================================
# 3. 模型评估（支付风控视角）
# ============================================================
print("\n" + "="*60)
print("模型评估（支付风控视角）")
print("="*60)

# 3.1 AUC
overall_auc = roc_auc_score(y, oof_preds)
print(f"\n整体 AUC: {overall_auc:.4f}")

# 3.2 KS值（你熟悉的指标）
from scipy.stats import ks_2samp
ks_stat = ks_2samp(oof_preds[y==1], oof_preds[y==0]).statistic
print(f"KS值: {ks_stat:.4f}")

# 3.3 Precision-Recall（支付风控最核心的指标）
# 面试点：为什么PR比ROC更重要？
# 因为在极度不均衡的数据中，ROC会过于乐观。
# 支付风控更关心：在你拦截的交易中，有多少真的是欺诈（Precision）
precision, recall, thresholds = precision_recall_curve(y, oof_preds)
ap_score = average_precision_score(y, oof_preds)
print(f"Average Precision (AP): {ap_score:.4f}")

# 3.4 不同阈值下的业务指标（这才是面试重点！）
print("\n--- 不同阈值下的业务表现 ---")
print(f"{'阈值':<10}{'拦截率':<12}{'精确率':<12}{'召回率':<12}{'F1':<10}{'业务含义'}")
print("-" * 70)

for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
    pred_labels = (oof_preds >= threshold).astype(int)
    tp = ((pred_labels == 1) & (y == 1)).sum()
    fp = ((pred_labels == 1) & (y == 0)).sum()
    fn = ((pred_labels == 0) & (y == 1)).sum()
    tn = ((pred_labels == 0) & (y == 0)).sum()

    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
    block_rate = (tp + fp) / len(y)

    # 业务含义解释
    if threshold <= 0.1:
        meaning = "宽松放行，漏放多"
    elif threshold <= 0.3:
        meaning = "平衡策略"
    elif threshold <= 0.5:
        meaning = "适中"
    elif threshold <= 0.7:
        meaning = "严格拦截"
    else:
        meaning = "极严格，误杀多"

    print(f"{threshold:<10.1f}{block_rate:<12.4f}{precision_val:<12.4f}{recall_val:<12.4f}{f1:<10.4f}{meaning}")

# ============================================================
# 4. 特征重要度分析
# ============================================================
print("\n--- Top 20 重要特征 ---")
mean_importance = feature_importance.groupby('feature')['importance'].mean()
top_features = mean_importance.sort_values(ascending=False).head(20)
for i, (feat, imp) in enumerate(top_features.items(), 1):
    print(f"{i:2d}. {feat:<30s} {imp:.0f}")

# 可视化
plt.figure(figsize=(10, 8))
top_features.sort_values().plot(kind='barh')
plt.title('Top 20 Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('./output/03_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 5. 保存模型评分
# ============================================================
train['fraud_score'] = oof_preds
train[['TransactionID', 'isFraud', 'fraud_score']].to_csv(
    './data/fraud_scores.csv', index=False
)
print("\n模型评分已保存到 ./data/fraud_scores.csv")

# ============================================================
# 面试讲述要点
# ============================================================
print("""
\n=== 面试讲述要点 ===

1. 为什么用LightGBM？
   - 处理高维稀疏特征效率高
   - 天然支持缺失值（支付数据缺失多）
   - 支持类别特征直接输入
   - 工业界支付风控主流选择

2. 样本不均衡怎么处理？
   - scale_pos_weight（首选，简单有效）
   - focal loss（进阶）
   - 不用SMOTE/过采样（数据量够大时没必要，还会引入噪声）

3. 为什么PR比ROC更重要？
   - 欺诈率只有3.5%，ROC的FPR分母太大，曲线会过于乐观
   - 业务关心的是：拦下来的里面有多少是真的（Precision）
   - 以及：真正的欺诈能抓住多少（Recall）

4. 模型评估不能只看AUC：
   - 必须看不同阈值下的Precision/Recall/拦截率
   - 最终阈值选择是业务决策，不是技术决策
   - 要和策略设计（下一步）联合看
""")
