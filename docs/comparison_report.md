# pytafast vs. R (TTR) 指标比对报告

本文档记录了 `pytafast`（基于 C 语言 `TA-Lib`）与 R 语言 `TTR` 包在计算金融指标时的对比结果。测试数据基于 `data/berkshire_1y.csv` 真实市场数据。

## 1. 总体结论

在测试的 **19 个核心指标**中，**16 个 (84%) 指标完全对齐**。

### 完全对齐的指标 (100% Match)
- **重叠研究**: SMA, EMA, WMA, TYPPRICE, WCLPRICE, MEDPRICE, AVGPRICE
- **动量指标**: RSI, MOM, ROC, CCI, MFI
- **波动率**: ATR, TRANGE
- **成交量**: OBV, AD

> **注意**: 为实现 RSI 和 ATR 的完全对齐，R 语言中必须设置 `maType = "EMA", wilder = TRUE`。

---

## 2. 存在差异的指标详细分析

### 2.1 MACD (Moving Average Convergence/Divergence)
*   **差异表现**: 在测试数据中，数值差异显著（Max Diff ~2495），匹配率仅约 33%。
*   **技术原因**:
    *   **不平稳期处理 (Unstable Period)**: TA-Lib 内部实现的 `TA_MACD` 遵循了更严格的“冷启动”消除逻辑。它会跳过开头的一部分数据以确保 EMA 进入稳定状态。
    *   **R 语言实现**: `TTR::MACD` 倾向于直接使用 `EMA(12) - EMA(26)`。
    *   **验证**: 如果在 Python 中手动计算 `pytafast.EMA(12) - pytafast.EMA(26)`，其结果与 R 完美对齐。
*   **评估**: 这种差异属于行业标准实现差异，不影响长期趋势判断。

### 2.2 SAR (Parabolic Stop and Reverse)
*   **差异表现**: 匹配率极低 (8.43%)。
*   **技术原因**:
    *   **加速因子 (AF) 逻辑**: TA-Lib 的 `TA_SAR` 在处理极值（Extreme Point）更新和加速因子的递增步长上与 TTR 存在算法细节差异。
    *   **初始化**: 两个库对于第一笔交易的基准价格选择不同。
*   **评估**: SAR 对初始条件高度敏感，不同库之间的差异是普遍现象。

### 2.3 Williams %R (WILLR)
*   **差异表现**: 数值完全不匹配（Max Diff 100）。
*   **技术原因**:
    *   **量纲定义**: 
        *   **TA-Lib**: 返回范围为 `[-100, 0]`。
        *   **TTR (WPR)**: 返回范围为 `[0, 1]`。
    *   **公式本质**: 两者公式本质相同，均为 `(Highest High - Close) / (Highest High - Lowest Low)` 的变体。
*   **评估**: 仅为输出格式差异，通过公式 `(WPR - 1) * 100` 即可完全转换对齐。

---

## 3. 迁移与使用建议

1.  **从 R 迁移到 Python**: 
    *   可以直接信任 `pytafast` 的绝大多数指标。
    *   如果需要与 R 结果绝对数值对齐，请在 Python 中手动通过基础指标（如 EMA）拼接复杂指标。
2.  **数值敏感性**: 在进行策略回测迁移时，建议至少保留 200 根 K 线以上的预热期，以消除不同库在不平稳期处理上的差异。

## 4. R (TTR/quantmod) 独有指标 vs. pytafast

通过对 `third_party/quantmod` 和 `third_party/TTR` 源码的审计，发现以下指标在 R 中有成熟实现，但目前在 `pytafast` (TA-Lib) 中缺失：

### 4.1 核心缺失指标
| 指标名称 | R 实现函数 | 功能说明 |
| :--- | :--- | :--- |
| **唐奇安通道** | `TTR::DonchianChannel` | 计算特定周期内的最高高价和最低低价。 |
| **肯特纳通道** | `TTR::keltnerChannels` | 基于 EMA 和 ATR 的波动率通道。 |
| **顾比复合移动平均** | `TTR::GMMA` | 结合短、长期多组 EMA 的趋势判断工具。 |
| **之字转向** | `TTR::ZigZag` | 过滤价格噪音，仅保留显著价格变动。 |
| **确定的事** | `TTR::KST` | 基于四个不同周期的 ROC 及其平滑值的动量振荡指标。 |
| **趋势检测指数** | `TTR::TDI` | 用于检测趋势的开始和结束。 |

### 4.2 源码逻辑对比发现
1.  **DonchianChannel (唐奇安通道)**:
    *   **R 逻辑**: 直接调用 `runMax(High, n)` 和 `runMin(Low, n)`。
    *   **pytafast 替代方案**: 虽然没有现成的 `DonchianChannel` 封装，但可以通过 `pytafast.MAX` 和 `pytafast.MIN` 手动组合。
2.  **ZigZag (之字转向)**:
    *   **R 逻辑**: 内部使用 C 语言递归处理价格极值，并结合 `na.approx` 进行线性插值。
    *   **pytafast 状态**: 缺失。由于涉及到非固定的 Lookback 和插值逻辑，TA-Lib 原生库未提供此功能。
3.  **统计定义差异 (STDDEV)**:
    *   **R (`runSD`)**: 默认除以 $N-1$（样本标准差）。
    *   **pytafast (`STDDEV`)**: 遵循 TA-Lib 标准，除以 $N$（总体标准差）。

## 5. 迁移建议

对于依赖 `ZigZag` 或 `GMMA` 等高级指标的 R 用户，在迁移到 `pytafast` 时，需要：
1.  **手动实现**: 利用 `pytafast` 提供的基础算子（如 EMA, MAX, MIN）重新构建这些复杂指标。
2.  **量纲转换**: 针对 `WILLR` 等指标，需注意 `[-100, 0]` 与 `[0, 1]` 的转换。
3.  **统计口径**: 在需要严格对齐标准差相关指标时，注意 $N$ 与 $N-1$ 的区别。

## 6. 复现方法

运行项目根目录下的自动化对比脚本：
```bash
./scripts/run_comparison.sh
```
该脚本会调用 `scripts/` 下的 Python 和 R 脚本，生成实时对比报告并自动清理临时数据。
