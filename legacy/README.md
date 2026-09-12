# 历史实现管理规则

历史实现目前继续保留在原路径。

`legacy/` 当前主要是一个逻辑边界，
并不意味着所有历史文件已经物理移动到本目录。

## 当前仍保留在原位置的历史区域

例如：

- `scripts/pilot_v2/`
- `scripts/mtpatcher_v3/`
- `scripts/mtpatcher_v4/`
- `scripts/mtpatcher_v5/`
- `scripts/mtpatcher_v6/`
- `scripts/mtpatcher_v7/`
- `scripts/mtpatcher_v8/`
- `scripts/mtpatcher_v9/`
- `scripts/mtpatcher_v10/`
- `scripts/mtpatcher_v11/`
- `scripts/mtpatcher_v14/`
- `scripts/mtpatcher_paper_repro/`
- `scripts/mtpatcher_paper_faithful_v2/`
- `scripts/mtpatcher_rq0/`
- `scripts/` 根目录中的旧 OPD / recovery / mechanism probes。

这些历史路径仍然可能用于：

- 重建历史结果；
- 来源追溯审计；
- 与早期 custom implementation 做比较；
- 复查已经被科研记录引用过的机制实验。

## 新正式实验规则

不要继续在历史版本目录中增加新的 canonical training infrastructure。

新的 框架原生 工作优先使用：

    configs/
    recipes/
    scripts/data/
    scripts/opd/
    scripts/eval/
    scripts/analysis/
    scripts/infra/
    scripts/定向/
    tests/
    manifests/

只有在确实有新的代码或资产需要时，才创建新的子目录。

不要继续创建：

    mtpatcher_v15/
    mtpatcher_v16/
    ...

这种纯时间版本目录作为新的主开发结构。

## 物理迁移规则

不要仅为了目录看起来整齐就移动历史文件。

只有满足以下条件时才考虑物理迁移：

1. 相关 active experiment 已经结束；
2. 路径依赖已经审计；
3. 历史 manifest 不再依赖旧路径；
4. 迁移本身由 Git 完整记录。

在此之前，历史代码保持原路径冻结。
