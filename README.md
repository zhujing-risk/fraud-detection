# 信用卡交易欺诈检测项目

## 项目背景
基于 Kaggle IEEE-CIS Fraud Detection 数据集，构建交易欺诈检测模型与风控策略体系。
模拟跨境电商支付风控场景，展示从数据分析→特征工程→建模→策略设计的全链路能力。

## 数据集
- Kaggle: https://www.kaggle.com/c/ieee-fraud-detection
- 训练集: ~590k 笔交易，欺诈率约 3.5%
- 特征: 交易金额、卡信息、设备信息、地址、时间等 400+ 维

## 项目结构
```
fraud_detection_project/
├── README.md
├── 01_eda_and_analysis.py          # 数据探索与欺诈模式分析
├── 02_feature_engineering.py        # 特征工程
├── 03_modeling.py                   # 模型训练与评估
├── 04_strategy_design.py            # 风控策略设计与模拟
└── 05_business_report.py            # 业务分析报告输出
```

## 面试讲述要点
1. 业务理解：欺诈率vs通过率的平衡
2. 特征工程：时间窗口聚合、设备关联、交易序列异常
3. 模型选择：为什么用LightGBM + 如何处理样本不均衡
4. 策略分层：高/中/低风险的差异化处置
5. 效果量化：预计减少欺诈损失X%，误拒率控制在Y%以内
