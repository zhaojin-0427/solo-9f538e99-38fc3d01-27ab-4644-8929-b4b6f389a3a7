"""嵌套重叠记录的整批更正回归验证(对应 MemoryError 修复)。

场景:
  - 单车道 2 万条重叠检测记录: 两台设备, 跑次各异,
    里程区间逐条嵌套、两端各错开 1 米(第 i 条为 [i, 2N-i));
  - 一段重铺, 检测时刻一半落在完工前、一半落在完工后;
  - 先录入, 再按原批号整批更正(整体平移 500 米、完工前后跑次互换)。

修复前: 重算为每个原子段物化全部有效 (设备, 跑次) 集合,
原子段数 x 每段跑次数为平方级 -> 256 MiB 地址空间下 MemoryError。
修复后: 增量维护有效集合, 内存 O(记录数), 收齐文件后 20 秒内提交成功。

运行:
  python scripts/nested_correction_check.py
  # 或模拟 256 MiB 地址空间限制:
  bash -c 'ulimit -v 262144; python scripts/nested_correction_check.py'
"""
from __future__ import annotations

import os
import resource
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import apply_batch, query_acceptance
from app.db import get_conn, init_db

LANE = "PERF-NESTED"
N = 20_000
REPAVE_T = 1_700_000_500
BUDGET_S = 20.0


def nested_rows(offset, swap):
    """第 i 条 [i+offset, 2N-i+offset), 两台设备交替, 跑次各异,
    检测时刻一半在完工前、一半在完工后(swap=True 时前后互换)。"""
    rows = []
    for i in range(N):
        dev = "DEV-A" if i % 2 == 0 else "DEV-B"
        before = (i < N // 2) != swap
        rows.append((i + offset, 2 * N - i + offset,
                     REPAVE_T - 10 if before else REPAVE_T + 10,
                     dev, f"RUN-{i:05d}"))
    return rows


def submit(kind, batch, rows, label):
    conn = get_conn()
    t0 = time.perf_counter()
    conn.execute("BEGIN IMMEDIATE")
    apply_batch(conn, LANE, kind, batch, rows, 1_700_100_000)
    conn.commit()
    conn.close()
    dt = time.perf_counter() - t0
    print(f"{label:<46} {dt:7.2f} s", flush=True)
    return dt


def main():
    os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "nested.db")
    init_db()

    submit("repave", "RP-1", [(N // 2, 3 * N // 2, REPAVE_T)],
           "upload repave (1 row)")
    submit("inspect", "JC-1", nested_rows(0, swap=False),
           f"upload inspect JC-1 ({N} nested rows)")
    dt = submit("inspect", "JC-1", nested_rows(500, swap=True),
                f"correct inspect JC-1 ({N} nested rows)")

    conn = get_conn()
    segs, _ = query_acceptance(conn, LANE, 0, 2 * N + 500, limit=10 ** 9)
    n_insp = conn.execute(
        "SELECT COUNT(*) FROM inspect_segments WHERE lane=?", (LANE,)
    ).fetchone()[0]
    conn.close()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"acceptance segments: {len(segs)}, inspect rows: {n_insp}, "
          f"peak RSS: {peak:.0f} MiB")

    assert n_insp == N, "更正后应只剩新批的 2 万行"
    assert not [s for s in segs if s["end_m"] <= 500], "旧批独有区段不应残留结果行"
    assert dt <= BUDGET_S, f"更正耗时 {dt:.1f}s 超过 {BUDGET_S}s 预算"
    print(f"OK: 更正 {dt:.2f}s <= {BUDGET_S}s, 结果校验通过")


if __name__ == "__main__":
    main()
