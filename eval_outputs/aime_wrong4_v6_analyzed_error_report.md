# 错因分析报告

- 样本：4
- 正确：2
- 错误：2
- 可定位覆盖率：100.0%

## 主错因分布

| 错因 | 数量 | 占错误比例 |
|---|---:|---:|
| verifier 误判 (`verifier_misjudgment`) | 2 | 100.0% |

## 典型样本

### verifier 误判

- `harp_1984_aime_10_957`：预测 `30`，期望 `119`；证据 `[{'signal': 'wrong_consensus_lock', 'answer': '30', 'support': 2}, {'signal': 'verifier_misjudgment', 'items': [{'kind': 'false_accept', 'answer': '30', 'role': 'constraint_interval_solver'}, {'kind': 'false_accept', 'answer': '30', 'role': 'algebra_cross_checker'}]}]`
- `harp_1984_aime_14_961`：预测 `18`，期望 `38`；证据 `[{'signal': 'wrong_consensus_lock', 'answer': '18', 'support': 4}, {'signal': 'verifier_misjudgment', 'items': [{'kind': 'false_accept', 'answer': '18', 'role': 'modular_solver'}]}]`
