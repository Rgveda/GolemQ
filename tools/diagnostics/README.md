# `tools/diagnostics/` —— 运行时诊断脚本

**不是测试，也不是库代码。** 这些脚本**连 MongoDB**，用来查看/清理**正在跑的系统的状态**
（心跳记录、互斥锁、实例 id）。它们以前混在 `GolemQ/test_cases/` 里，
2026-10-10 分出来 —— 那些脚本本来就不参与 `unittest` 发现（它只收 `test_*.py`），
但摆在一起会让「测试目录里有多少真测试」看不出来。

## 与 `tools/` 的关系

| 目录 | 装什么 | 生命周期 |
|:--|:--|:--|
| `tools/` | **要反复用**的仓库开发工具（`gen_api_index.py` 每次改完重生成、`fix_ruflo_hooks.py` 每次 `ruflo init` 后重跑、审计/探测脚本）| **长期维护** |
| `tools/diagnostics/`（这里）| **排查用**的一次性脚本 | **用完即弃** |

## 这一族服务的是哪个缺陷

`MIGRATION_STATUS.md` 第六节记的 **`HeartbeatModule.mutex()` 非原子 check-then-act**：
唯一索引建在 `(module_name, instance_id)`，而 `instance_id` 是 **per-process 的 sha256**
⇒ **两个进程各写各的 id，索引永不冲突、永远拦不住第二个进程**。**该缺陷尚未修**。

所以这些脚本**不是垃圾**：它们编码的是「怎么查、怎么清」那个状态。等缺陷修好，本目录即可删除
（git 历史留着，随时能捞回来）。

| 脚本 | 干什么 |
|:--|:--|
| `check_heartbeat.py` | 看心跳记录（谁在跑、超时没）|
| `check_running_modules.py` | 列正在运行的模块实例 |
| `verify_heartbeat.py` | 复核心跳/超时判据 |
| `debug_instance_id.py` | 看 `instance_id` 是怎么生的（那个 sha256）|
| `cleanup_timeout_instance.py` | 清掉超时实例的锁记录 |
| `direct_cleanup.py` | 同上，直接按 `_id` 清 |
| `final_cleanup.py` | ⚠️ 名字里的 `final` 说明这是**当时调参的最后一版** —— 与上面两个是同一件事的三次迭代，**别再新增第四个**，要改就改现有的 |

## 跑法

```bash
python tools/diagnostics/check_heartbeat.py
```

⚠️ **它们会连真实的 `golemq` 库**（心跳/签到/关注列表在那个库）。
`cleanup_*` 那几个是**写操作**，动手前先跑只读的那几个看清状态。
