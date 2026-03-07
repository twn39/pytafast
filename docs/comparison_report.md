# pytafast vs. R (TTR) 指标比对与迁移报告

本文档记录了 `pytafast` 与 R 语言 `TTR`/`quantmod` 包在计算金融指标时的对比结果。测试基于多源市场数据（A 股、美股、韩股），涵盖了 150+ 个指标。

## 1. 总体结论

`pytafast` 现已实现了对 R 语言 `TTR` 包核心功能的**深度覆盖**。

*   **完全对齐的指标 (100% Match)**: SMA, EMA, WMA, **HMA**, **ALMA**, **ZLEMA**, **EVWMA**, **DonchianChannel**, **keltnerChannels**, **CMF**, **DPO**, **VHF**, **SNR**, RSI, MOM, ROC, CCI, MFI, ATR, TRANGE, OBV, AD 等。
*   **高度对齐但有微小差异**: **ZigZag** (98%), **KST**, **SMI**, MACD, SAR。
*   **量纲/定义差异**: WILLR ([-100, 0] vs [0, 1]), STDDEV (N vs N-1)。

---

## 2. 核心指标对齐与差异分析

### 2.1 移动平均变体 (ALMA, ZLEMA, EVWMA)
*   **ALMA**: 已完全对齐 R 语言的 `floor(offset * (n-1))` 高斯权重分布。
*   **ZLEMA**: 采用了 TTR 风格的**分数延迟插值**算法，解决了简单线性减法在非整数周期下的精度问题。
*   **EVWMA**: 对齐了基于 $n$ 周期成交量总和的递归更新逻辑，保证了“弹性”系数的一致性。

### 2.2 MACD & SAR (冷启动差异)
*   **MACD**: `pytafast` (TA-Lib) 内部有严格的不平稳期 (Unstable Period) 处理逻辑，而 R 语言通常直接从第一组可用数据开始计算。
*   **SAR**: 加速因子 (AF) 的更新触发点和极值 (EP) 的初始化策略不同，导致该指标在初期极度敏感且难以完全对齐。

### 2.3 ZigZag (功能增强)
*   **差异**: 匹配率约 98%。
*   **原因**: `pytafast` 的 C++ 实现包含**自动延伸至最新价格**的逻辑（Linear Extension），而 R 在最后一个确定极值点之后会保持 `NA`。
*   **结论**: `pytafast` 版本更适合实时看板展示。

---

## 3. R 原生指标迁移清单

通过对 `third_party` 源码的审计，以下原属 R 特有的指标现已进入 `pytafast`：

| 指标名称 | pytafast 调用 | 实现层级 | 对齐状态 |
| :--- | :--- | :--- | :--- |
| **Arnaud Legoux MA** | `ALMA()` | C++/GSL | 100% |
| **Elastic Volume WMA**| `EVWMA()` | C++/GSL | 100% |
| **Zero Lag EMA** | `ZLEMA()` | C++/GSL | 100% |
| **之字转向** | `ZIGZAG()` | C++/GSL | 98% (末端延伸) |
| **唐奇安通道** | `DonchianChannel()` | Python 组合 | 100% |
| **肯特纳通道** | `keltnerChannels()` | Python 组合 | 100% |
| **蔡金资金流量** | `CMF()` | Python 组合 | 100% |
| **确定的事** | `KST()` | Python 组合 | 算法对齐 (Discrete ROC) |

---

## 4. 工程安全性与性能

### 4.1 GSL 安全性增强
C++ 扩展层全面引入了 **GSL (Guideline Support Library)**：
*   使用 `gsl::span` 替代原始指针，确保 `ZigZag` 和 `ALMA` 等循环密集型算法的内存访问安全。
*   使用 `gsl::narrow` 进行类型安全转换，防止 `size_t` 溢出。

### 4.2 高性能计算
所有迁移后的复杂指标均支持：
*   **GIL 释放**：计算期间不占用 Python 全局锁。
*   **零拷贝**：直接在 `numpy` 内存缓冲区上操作。
*   **NaN 鲁棒性**：内部自动处理 `ROC` 或 `EMA` 产生的引导 `NaN`，确保级联计算不失效。

---

## 5. 迁移建议

1.  **统计口径**: 计算 `STDDEV` 或 `VAR` 时，注意 `pytafast` 使用总体标准差 (除以 $N$)。
2.  **不平稳期**: 策略回测时建议预留至少 200 根 K 线作为“温缸”数据，以消除库间初始化差异。
3.  **多源验证**: 运行 `./scripts/run_comparison.sh <data_path>` 可对任意数据集执行自动化对齐校验。

---

## 6. 复现方法

```bash
# 执行全量自动化验证报告
./scripts/run_comparison.sh data/nasdaq100_2025_now.csv
```
