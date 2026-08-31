"""
04 - 风控策略设计与模拟
目标：基于模型分输出，设计分层风控策略，模拟真实业务决策
面试价值：这是最能体现"风控策略"岗位能力的部分 —— 不只是建模，而是设计策略

核心思路（对标真实支付风控）：
- 高分（低风险）→ 直接放行（保体验）
- 中分 → 挑战验证（3DS/短信验证）
- 低分（高风险）→ 直接拦截
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 加载模型评分
# ============================================================
scores = pd.read_csv('./data/fraud_scores.csv')
print(f"总交易量: {len(scores):,}")
print(f"欺诈率: {scores['isFraud'].mean():.4f}")
print(f"模型分分布:\n{scores['fraud_score'].describe()}")

# ============================================================
# 2. 分层策略设计（核心！）
# ============================================================
print("\n" + "="*60)
print("分层风控策略设计")
print("="*60)

# 策略参数（实际业务中这些阈值通过AB测试确定）
THRESHOLD_HIGH_RISK = 0.7    # 高于此分 → 直接拦截
THRESHOLD_MED_RISK = 0.3     # 高于此分且低于高风险 → 挑战验证
# 低于0.3 → 直接放行

# 模拟挑战验证的通过率
# 面试点：挑战验证不是100%拦截，正常用户会通过验证
CHALLENGE_PASS_RATE_NORMAL = 0.85   # 正常用户通过验证的概率
CHALLENGE_PASS_RATE_FRAUD = 0.15    # 欺诈者通过验证的概率

def apply_strategy(df, high_threshold, med_threshold):
    """模拟风控策略执行"""
    df = df.copy()

    # 分层
    df['risk_level'] = 'low'
    df.loc[df['fraud_score'] >= med_threshold, 'risk_level'] = 'medium'
    df.loc[df['fraud_score'] >= high_threshold, 'risk_level'] = 'high'

    # 决策
    df['decision'] = 'approve'  # 默认放行

    # 高风险直接拦截
    df.loc[df['risk_level'] == 'high', 'decision'] = 'block'

    # 中风险进入挑战验证
    medium_mask = df['risk_level'] == 'medium'
    medium_fraud = medium_mask & (df['isFraud'] == 1)
    medium_normal = medium_mask & (df['isFraud'] == 0)

    # 模拟验证结果
    np.random.seed(42)
    # 正常用户大部分通过验证
    df.loc[medium_normal, 'decision'] = np.where(
        np.random.random(medium_normal.sum()) < CHALLENGE_PASS_RATE_NORMAL,
        'approve', 'block'
    )
    # 欺诈者大部分无法通过验证
    df.loc[medium_fraud, 'decision'] = np.where(
        np.random.random(medium_fraud.sum()) < CHALLENGE_PASS_RATE_FRAUD,
        'approve', 'block'
    )

    return df

result = apply_strategy(scores, THRESHOLD_HIGH_RISK, THRESHOLD_MED_RISK)

# ============================================================
# 3. 策略效果评估
# ============================================================
print("\n--- 各风险层级分布 ---")
level_stats = result.groupby('risk_level').agg(
    交易量=('isFraud', 'count'),
    欺诈量=('isFraud', 'sum'),
    欺诈率=('isFraud', 'mean'),
    平均分=('fraud_score', 'mean')
).round(4)
level_stats['占比'] = (level_stats['交易量'] / len(result)).round(4)
print(level_stats)

print("\n--- 策略整体效果 ---")
total_fraud = result['isFraud'].sum()
blocked_fraud = ((result['decision'] == 'block') & (result['isFraud'] == 1)).sum()
blocked_normal = ((result['decision'] == 'block') & (result['isFraud'] == 0)).sum()
approved_fraud = ((result['decision'] == 'approve') & (result['isFraud'] == 1)).sum()

fraud_catch_rate = blocked_fraud / total_fraud  # 欺诈拦截率（Recall）
false_block_rate = blocked_normal / (result['isFraud'] == 0).sum()  # 误拒率
overall_block_rate = (result['decision'] == 'block').sum() / len(result)  # 总拦截率
precision = blocked_fraud / (result['decision'] == 'block').sum()  # 拦截精确率

print(f"欺诈拦截率（Recall）: {fraud_catch_rate:.4f} ({fraud_catch_rate*100:.1f}%)")
print(f"误拒率（正常交易被拦）: {false_block_rate:.4f} ({false_block_rate*100:.2f}%)")
print(f"拦截精确率（Precision）: {precision:.4f} ({precision*100:.1f}%)")
print(f"总拦截率: {overall_block_rate:.4f} ({overall_block_rate*100:.2f}%)")
print(f"通过率: {1-overall_block_rate:.4f} ({(1-overall_block_rate)*100:.2f}%)")
print(f"漏过欺诈: {approved_fraud}笔 / {total_fraud}笔")

# ============================================================
# 4. 经济损失测算（面试加分项！）
# ============================================================
print("\n" + "="*60)
print("经济损失测算（ROI分析）")
print("="*60)

# 假设参数（面试时可以说"这些参数在真实业务中由财务提供"）
AVG_FRAUD_LOSS = 150        # 每笔欺诈平均损失（美元）
CHARGEBACK_FEE = 25         # 每笔拒付手续费
CHALLENGE_COST = 0.5        # 每次挑战验证的成本（短信/3DS通道费）
FALSE_BLOCK_LOSS = 30       # 每笔误拒的机会成本（用户流失）

# 无风控的损失
no_strategy_loss = total_fraud * (AVG_FRAUD_LOSS + CHARGEBACK_FEE)
print(f"\n无风控场景总损失: ${no_strategy_loss:,.0f}")

# 有风控的损失
fraud_loss = approved_fraud * (AVG_FRAUD_LOSS + CHARGEBACK_FEE)  # 漏过的欺诈
false_block_cost = blocked_normal * FALSE_BLOCK_LOSS              # 误拒成本
challenge_cost = (result['risk_level'] == 'medium').sum() * CHALLENGE_COST  # 验证成本
total_cost_with_strategy = fraud_loss + false_block_cost + challenge_cost

print(f"有风控场景:")
print(f"  - 漏过欺诈损失: ${fraud_loss:,.0f}")
print(f"  - 误拒机会成本: ${false_block_cost:,.0f}")
print(f"  - 验证通道成本: ${challenge_cost:,.0f}")
print(f"  - 总成本: ${total_cost_with_strategy:,.0f}")
print(f"\n风控净收益: ${no_strategy_loss - total_cost_with_strategy:,.0f}")
print(f"损失降幅: {(1 - total_cost_with_strategy/no_strategy_loss)*100:.1f}%")

# ============================================================
# 5. 策略AB实验设计（展示实验思维）
# ============================================================
print("\n" + "="*60)
print("AB实验设计方案")
print("="*60)
print("""
实验目标：验证新分层策略相比现有策略是否能同时降低欺诈率和误拒率

实验设计：
┌─────────────────────────────────────────────────┐
│ 对照组(50%流量)：当前规则引擎（固定规则拦截）    │
│ 实验组(50%流量)：模型分+分层策略                 │
└─────────────────────────────────────────────────┘

核心观测指标：
- 主指标：欺诈率（实验组 < 对照组 → 策略有效）
- 护栏指标：通过率（实验组 ≥ 对照组 → 体验不恶化）
- 次要指标：人工审核率、挑战验证触发率、拒付率

实验周期：14天（覆盖完整的拒付回报周期）

统计方案：
- 显著性水平 α = 0.05
- 统计功效 1-β = 0.8
- 最小可检测效应：欺诈率相对下降 10%

风险兜底：
- 实验组欺诈率超过对照组150%时自动熔断
- 每日人工review高风险拦截case（前3天全量review）
""")

# ============================================================
# 6. 不同阈值的敏感性分析
# ============================================================
print("\n--- 阈值敏感性分析 ---")
print(f"{'高风险阈值':<12}{'中风险阈值':<12}{'拦截率':<10}{'误拒率':<10}{'欺诈拦截率':<12}")
print("-" * 56)

for high_t in [0.5, 0.6, 0.7, 0.8, 0.9]:
    for med_t in [0.1, 0.2, 0.3]:
        r = apply_strategy(scores, high_t, med_t)
        block_rate = (r['decision'] == 'block').sum() / len(r)
        false_block = ((r['decision']=='block') & (r['isFraud']==0)).sum() / (r['isFraud']==0).sum()
        fraud_catch = ((r['decision']=='block') & (r['isFraud']==1)).sum() / r['isFraud'].sum()
        print(f"{high_t:<12.1f}{med_t:<12.1f}{block_rate:<10.4f}{false_block:<10.4f}{fraud_catch:<12.4f}")

print("""
\n=== 面试讲述要点 ===

1. 策略设计的核心逻辑：
   "不是所有交易都需要同样的风控强度 —— 好客户少打扰，坏客户强拦截"
   这和你在信贷做的差异化策略完全一样

2. 三层策略的业务含义：
   - 放行：用户无感，体验最好
   - 挑战验证：轻度摩擦，过滤大部分欺诈
   - 强拦截：直接拒绝，用于高置信欺诈

3. 阈值怎么定？
   "不是拍脑袋，而是通过AB实验+损益测算来确定最优阈值"
   → 这就是你之前做AB实验的能力迁移

4. 经济损失测算：
   "每条规则都有成本（误杀好人）和收益（拦住坏人）"
   → 和你在度小满做的损益测算完全一样的方法论
""")
