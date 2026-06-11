# ReTune SFB 295 — WP3 Project Context

## 项目概述

**项目名称**：SFB ReTune 295，INF (Information Infrastructure)  
**负责人**：Hanwei Liu  
**目标**：为 SFB 所有项目提供统一的容器化神经科学分析框架

## 三个子工作包

| 子项目 | 核心内容 | 容器 |
|--------|---------|------|
| **WP3a** | 多模态时间序列因果分析 | Docker I（功能） + Docker II（结构） |
| **WP3b** | 数据驱动建模，疾病亚期识别 | Docker III |
| **WP3c** | TVB 全脑仿真，in silico DBS，BIDS + DataLad | Apptainer（HPC兼容）|

## 数据模态

**当前主要数据模态**：EEG, LFP, MEG, fMRI  
**注**：其他模态（NIRS, ECG, EMG, DWI 等）在框架设计中预留接口，但目前分析以上述三种为主

## 参考数据集

目前尚无参考数据集。分析结果暂不做跨数据集对比验证，待后续与 S01（人类 atlas）或 S02（小鼠 atlas）对接时在此更新。

## 目标用户

从工程师到临床医生的多学科团队——框架设计需考虑可用性，不只是技术性。

## 技术栈总览

```
Python:    MNE, IDTxl, frites, bctpy, pymc, statsmodels, sklearn, semopy
TVB:       tvb-library, tvb-data, TVB-O 本体论
BIDS:      MNE-BIDS, bids-validator, BIDS 计算模型扩展（草案）
容器:      Docker, Apptainer/Singularity
数据管理:  DataLad, GIN, EBRAINS Knowledge Graph
```

## 可复现性要求

- 所有工作流使用容器封装，确保系统无关
- DataLad 追踪数据来源和处理溯源
- 遵循 FAIR 原则（Findable, Accessible, Interoperable, Reusable）
- 计算模型遵循 TVB-O 本体论标准
- 发布模型在 EBRAINS 上可发现，开源许可证

## 可用 Agents

- `@container-engineer`：容器构建和维护
- `@connectivity-analyst`：WP3a 因果分析实现
- `@modeling-analyst`：WP3b 统计建模
- `@tvb-bids-engineer`：WP3c TVB 仿真 + BIDS 管理
- `@research-documenter`：文献、日志、文档（用 Haiku 省 token）

## 工作目录结构

```
project/
├── CLAUDE.md               ← 本文件
├── CHANGES.md              ← 版本变更记录
├── containers/
│   ├── docker-I/
│   ├── docker-II/
│   ├── docker-III/
│   └── apptainer/
├── analysis/
│   ├── WP3a_causality/
│   ├── WP3b_modeling/
│   └── WP3c_tvb/
├── data/                   ← DataLad 管理
├── logs/                   ← 实验日志
├── references/             ← 文献整理
├── reports/                ← 进度报告
└── docs/                   ← 技术文档
```
## 输出文件规范

Claude Code 生成的所有报告、计划、日志文件统一放到：
`.claude/outputs/` 目录下，文件名格式：`YYYY-MM-DD_[类型]_[描述].md`

例如：
- `.claude/outputs/2026-04-02_cleanup_plan.md`
- `.claude/outputs/2026-04-02_cleanup_report.md`

