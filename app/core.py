"""核心业务逻辑。

- CSV 解析与校验(整批校验, 任何一行错误则整批拒绝)
- 区间扫描(sweep line)计算交验结果, 只按区间端点切分, 不逐米展开
- 批次应用: 原始批次与交验结果在同一事务中生效; 更正只重算旧批和新批涉及的米段

交验规则:
  每处(每个原子区间)需要 NEEDED_DEVICES 台不同设备的有效跑次覆盖才算合格;
  已重铺的位置, 只认最近一次完工时刻(threshold)之后检测的跑次。
"""
from __future__ import annotations

import csv
import heapq
import io
from collections import defaultdict

NEEDED_DEVICES = 2  # 每处合格所需的不同设备台数

REPAVE_COLUMNS = ["start_m", "end_m", "completed_at"]
INSPECT_COLUMNS = ["start_m", "end_m", "measured_at", "device_id", "run_id"]

MAX_ROWS_PER_UPLOAD = 1_000_000
MAX_FIELD_LEN = 64

ACC_COLUMNS = (
    "lane,start_m,end_m,threshold,device_count,missing,device1,run1,device2,run2"
)
_ACC_ATTRS = ("threshold", "device_count", "missing", "device1", "run1", "device2", "run2")


class CsvError(ValueError):
    """CSV 内容不合法, 整批拒绝。"""


# ---------------------------------------------------------------- CSV 解析

def _to_int(raw, name, lineno, errors):
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        errors.append(f"第{lineno}行: {name} 不是整数: {raw!r}")
        return None


def _to_text(raw, name, lineno, errors):
    s = (raw or "").strip()
    if not s:
        errors.append(f"第{lineno}行: {name} 不能为空")
        return None
    if len(s) > MAX_FIELD_LEN:
        errors.append(f"第{lineno}行: {name} 超过 {MAX_FIELD_LEN} 个字符")
        return None
    return s


def _check_span(s, e, lineno, errors):
    if s is None or e is None:
        return False
    if s < 0 or e <= s:
        errors.append(f"第{lineno}行: 需满足 0 <= start_m < end_m, 实际为 [{s}, {e})")
        return False
    return True


def _parse_csv(text, columns, build_row):
    reader = csv.reader(io.StringIO(text))
    rows, errors = [], []
    idx = None
    for lineno, rec in enumerate(reader, start=1):
        if not rec or all(not c.strip() for c in rec):
            continue  # 跳过空行
        if idx is None:
            header = [c.strip() for c in rec]
            missing = [c for c in columns if c not in header]
            if missing:
                raise CsvError(f"CSV 缺少列 {missing}; 需要的列: {columns}")
            idx = {c: header.index(c) for c in columns}
            continue
        if len(rows) >= MAX_ROWS_PER_UPLOAD:
            errors.append(f"超过单次上传最大行数 {MAX_ROWS_PER_UPLOAD}")
            break
        row = build_row(rec, idx, lineno, errors)
        if row is not None:
            rows.append(row)
        if len(errors) >= 20:
            errors.append("错误过多, 仅显示前 20 条")
            break
    if idx is None:
        raise CsvError("CSV 为空或缺少表头")
    if errors:
        raise CsvError("; ".join(errors))
    return rows


def _repave_row(rec, idx, lineno, errors):
    get = lambda c: rec[idx[c]] if idx[c] < len(rec) else ""
    s = _to_int(get("start_m"), "start_m", lineno, errors)
    e = _to_int(get("end_m"), "end_m", lineno, errors)
    t = _to_int(get("completed_at"), "completed_at", lineno, errors)
    if t is not None and t < 0:
        errors.append(f"第{lineno}行: completed_at 需 >= 0")
        t = None
    if not _check_span(s, e, lineno, errors) or t is None:
        return None
    return (s, e, t)


def _inspect_row(rec, idx, lineno, errors):
    get = lambda c: rec[idx[c]] if idx[c] < len(rec) else ""
    s = _to_int(get("start_m"), "start_m", lineno, errors)
    e = _to_int(get("end_m"), "end_m", lineno, errors)
    t = _to_int(get("measured_at"), "measured_at", lineno, errors)
    d = _to_text(get("device_id"), "device_id", lineno, errors)
    r = _to_text(get("run_id"), "run_id", lineno, errors)
    if t is not None and t < 0:
        errors.append(f"第{lineno}行: measured_at 需 >= 0")
        t = None
    if not _check_span(s, e, lineno, errors) or None in (t, d, r):
        return None
    return (s, e, t, d, r)


def parse_repave_csv(text):
    """返回 [(start_m, end_m, completed_at), ...]"""
    return _parse_csv(text, REPAVE_COLUMNS, _repave_row)


def parse_inspect_csv(text):
    """返回 [(start_m, end_m, measured_at, device_id, run_id), ...]"""
    return _parse_csv(text, INSPECT_COLUMNS, _inspect_row)


# ---------------------------------------------------------------- 区间扫描

def sweep_segments(repaves, inspects, lo, hi):
    """在 [lo, hi) 上扫描计算交验状态, 不逐米展开。

    repaves:  可迭代的 (start, end, completed_at)
    inspects: 可迭代的 (start, end, measured_at, device_id, run_id)
    返回合并后的段列表 [(start, end, threshold, device_count, d1, r1, d2, r2)]:
    段内 最近一次重铺完工时刻(threshold, 无重铺为 0) 与有效 (设备, 跑次) 集合
    恒定; d1/d2 为按设备号排序前两台有有效跑次的设备(不足则 None), r1/r2 为
    该设备最小的有效跑次号, 且这些跑次都覆盖整段。

    有效 (设备, 跑次) 集合按端点事件增量维护(堆 + 懒删除), 每个端点 O(log)
    更新, 每个原子段只提取 O(设备数) 的展示状态, 不为每个原子段物化跑次集合:
    内存 O(记录数), 时间 O(记录数 · log 记录数)。2 万条逐条嵌套的重叠记录
    (原子段数 × 每段跑次数 = 平方级)也不会撑爆内存。
    """
    # 每个端点位置: (重铺进入, 重铺离开, 检测进入, 检测离开)
    events = defaultdict(lambda: ([], [], [], []))
    for s, e, t in repaves:
        s = lo if s < lo else s
        e = hi if e > hi else e
        if s < e:
            events[s][0].append(t)
            events[e][1].append(t)
    for s, e, t, d, r in inspects:
        s = lo if s < lo else s
        e = hi if e > hi else e
        if s < e:
            events[s][2].append((d, r, t))
            events[e][3].append((d, r, t))
    events[lo]  # 端点哨兵: 保证扫描覆盖整个 [lo, hi)
    events[hi]

    rep_active = {}   # completed_at -> 覆盖次数
    rep_heap = []     # 最大堆(存负值), 懒删除
    thr = 0           # 当前最近一次重铺完工时刻(无活动重铺为 0)

    tcount = {}              # (device, run) -> {measured_at: 覆盖次数}
    theap = {}               # (device, run) -> measured_at 最大堆(存负值), 懒删除
    valid = set()            # 当前有效 (device, run): 有活动记录且 measured_at > thr
    valid_runs = defaultdict(set)   # device -> 有效跑次号集合
    run_heap = defaultdict(list)    # device -> 跑次号最小堆, 懒删除
    dev_valid = set()        # 有有效跑次的设备
    dev_token = {}           # device -> 当前有效代次(失效再生效时递增)
    dev_heap = []            # (device, 代次) 最小堆, 过期代次懒删除
    changed = True  # 上一端点处 (threshold, 有效跑次集合) 是否发生变化

    def bump(cnt, key, delta):
        v = cnt.get(key, 0) + delta
        if v:
            cnt[key] = v
        else:
            cnt.pop(key, None)

    def max_t(key):
        """该 (设备, 跑次) 活动记录的最大检测时刻; 无活动记录为 None。"""
        h = theap.get(key)
        if not h:
            return None
        tc = tcount[key]
        while h:
            t = -h[0]
            if tc.get(t, 0) > 0:
                return t
            heapq.heappop(h)
        return None

    def set_valid(d, r, now):
        """按当前 thr 更新一个 (设备, 跑次) 的有效性, 同步派生结构。"""
        nonlocal changed
        if now == ((d, r) in valid):
            return
        changed = True
        if now:
            valid.add((d, r))
            runs = valid_runs[d]
            if not runs:
                dev_valid.add(d)
                dev_token[d] = dev_token.get(d, 0) + 1
                heapq.heappush(dev_heap, (d, dev_token[d]))
            runs.add(r)
            heapq.heappush(run_heap[d], r)
        else:
            valid.discard((d, r))
            runs = valid_runs[d]
            runs.discard(r)
            if not runs:
                dev_valid.discard(d)

    def refresh(d, r):
        m = max_t((d, r))
        set_valid(d, r, m is not None and m > thr)

    def clean_dev_top():
        """弹出已失效或已过期的设备堆顶(懒删除)。"""
        while dev_heap:
            d, tok = dev_heap[0]
            if d in dev_valid and dev_token.get(d) == tok:
                break
            heapq.heappop(dev_heap)

    def min_run(d):
        """设备 d 最小的有效跑次号, 堆顶懒删除。"""
        h, vr = run_heap[d], valid_runs[d]
        while h and h[0] not in vr:
            heapq.heappop(h)
        return h[0] if h else None

    merged = []
    open_seg = None   # [start, threshold, device_count, d1, r1, d2, r2]
    prev = None
    for pos in sorted(events):
        if prev is not None and pos > prev:
            if open_seg is not None and not changed:
                pass  # (threshold, 有效跑次集合) 与上一原子段相同, 延伸当前段
            else:
                if open_seg is not None:
                    merged.append((open_seg[0], prev, *open_seg[1:]))
                clean_dev_top()
                d1 = dev_heap[0][0] if dev_heap else None
                d2 = None
                if d1 is not None:
                    top1 = heapq.heappop(dev_heap)  # 临时取下 d1 看次小设备
                    clean_dev_top()
                    d2 = dev_heap[0][0] if dev_heap else None
                    heapq.heappush(dev_heap, top1)  # 放回, 供后续原子段使用
                open_seg = [
                    prev, thr, len(dev_valid), d1,
                    min_run(d1) if d1 is not None else None,
                    d2,
                    min_run(d2) if d2 is not None else None,
                ]
        # 处理 pos 处事件
        changed = False
        r_add, r_del, i_add, i_del = events[pos]
        for t in r_del:
            bump(rep_active, t, -1)
        for t in r_add:
            bump(rep_active, t, 1)
            heapq.heappush(rep_heap, -t)
        new_thr = 0
        while rep_heap:
            top = -rep_heap[0]
            if rep_active.get(top, 0) > 0:
                new_thr = top
                break
            heapq.heappop(rep_heap)
        if new_thr != thr:
            thr = new_thr
            changed = True
            for d, r in list(tcount):
                refresh(d, r)
        # 检测记录进出: 先更新计数, 再对涉及的 (设备, 跑次) 统一重判;
        # 同一端点先离后进的同一跑次净效果为零, 不会误切段
        touched = set()
        for d, r, t in i_del:
            key = (d, r)
            tc = tcount[key]
            v = tc.get(t, 0) - 1
            if v > 0:
                tc[t] = v
            else:
                del tc[t]
                if not tc:
                    del tcount[key]
                    theap.pop(key, None)
            touched.add(key)
        for d, r, t in i_add:
            key = (d, r)
            tc = tcount.setdefault(key, {})
            tc[t] = tc.get(t, 0) + 1
            heapq.heappush(theap.setdefault(key, []), -t)
            touched.add(key)
        for d, r in touched:
            refresh(d, r)
        prev = pos
    if open_seg is not None:
        merged.append((open_seg[0], prev, *open_seg[1:]))
    return merged


def segment_rows(lane, merged):
    """把扫描结果转成 acceptance 表行。"""
    return [
        (lane, s, e, thr, count, max(0, NEEDED_DEVICES - count), d1, r1, d2, r2)
        for s, e, thr, count, d1, r1, d2, r2 in merged
    ]


# ---------------------------------------------------------------- 交验结果落库

def _insert_acc(cur, row):
    cur.execute(f"INSERT INTO acceptance({ACC_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?)", row)


def merge_ranges(ranges):
    """合并重叠或相接的区间, 得到互不相交的重算组件。"""
    out = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out if s < e]


def _merge_boundary(cur, lane, pos):
    """重算窗口边界处, 若左右两段属性完全相同则合并, 避免产生碎段。"""
    left = cur.execute(
        "SELECT rowid,* FROM acceptance WHERE lane=? AND end_m=?", (lane, pos)
    ).fetchall()
    right = cur.execute(
        "SELECT rowid,* FROM acceptance WHERE lane=? AND start_m=?", (lane, pos)
    ).fetchall()
    if len(left) != 1 or len(right) != 1:
        return
    L, R = left[0], right[0]
    if all(L[k] == R[k] for k in _ACC_ATTRS):
        cur.execute(
            "DELETE FROM acceptance WHERE rowid IN (?,?)", (L["rowid"], R["rowid"])
        )
        _insert_acc(
            cur, (lane, L["start_m"], R["end_m"]) + tuple(L[k] for k in _ACC_ATTRS)
        )


def _clip_acceptance(cur, lane, a, b):
    """删除/裁剪与 [a, b) 相交的旧结果行, 窗口外部分保留。"""
    for row in cur.execute(
        "SELECT rowid,* FROM acceptance WHERE lane=? AND start_m<? AND end_m>?",
        (lane, b, a),
    ).fetchall():
        cur.execute("DELETE FROM acceptance WHERE rowid=?", (row["rowid"],))
        attrs = tuple(row[k] for k in _ACC_ATTRS)
        if row["start_m"] < a:
            _insert_acc(cur, (lane, row["start_m"], a) + attrs)
        if row["end_m"] > b:
            _insert_acc(cur, (lane, b, row["end_m"]) + attrs)


def _assign(components, rows):
    """把区间行分配给各有序组件(每组件命中的行 = 与该组件相交的行)。

    行与组件均按起点有序, 单趟扫描完成, 避免每个组件各做一次范围查询。
    """
    rows = sorted(rows, key=lambda r: r[0])
    out, active = [], []
    ri = 0
    for a, b in components:
        while ri < len(rows) and rows[ri][0] < b:
            active.append(rows[ri])
            ri += 1
        active = [r for r in active if r[1] > a]
        out.append(list(active))
    return out


def recompute_components(conn, lane, components):
    """重算一组互不相交、按起点升序的米段组件。

    交验状态是逐点函数, 只取决于覆盖该点的记录, 因此组件外的既有结果
    不受影响, 直接保留; 结果只生成在组件内有数据(重铺或检测)的连通米段上,
    段内无检测覆盖的空洞体现为"缺 2 台"的段, 完全无数据的区域不产生结果行。
    """
    if not components:
        return
    lo, hi = components[0][0], components[-1][1]
    repaves = [
        (r[0], r[1], r[2])
        for r in conn.execute(
            "SELECT start_m,end_m,completed_at FROM repave_segments "
            "WHERE lane=? AND start_m<? AND end_m>?",
            (lane, hi, lo),
        )
    ]
    inspects = [
        (r[0], r[1], r[2], r[3], r[4])
        for r in conn.execute(
            "SELECT start_m,end_m,measured_at,device_id,run_id FROM inspect_segments "
            "WHERE lane=? AND start_m<? AND end_m>?",
            (lane, hi, lo),
        )
    ]
    rep_sets = _assign(components, repaves)
    insp_sets = _assign(components, inspects)
    cur = conn.cursor()
    for (a, b), rep, insp in zip(components, rep_sets, insp_sets):
        _clip_acceptance(cur, lane, a, b)
        # 组件内数据的连通米段(裁剪到组件内), 逐段扫描并写入新结果
        hulls = merge_ranges(
            [(max(s, a), min(e, b)) for s, e, *_ in rep]
            + [(max(s, a), min(e, b)) for s, e, *_ in insp]
        )
        for hlo, hhi in hulls:
            for row in segment_rows(lane, sweep_segments(rep, insp, hlo, hhi)):
                _insert_acc(cur, row)
            # 与组件外相同属性的旧段合并, 避免碎段
            _merge_boundary(cur, lane, hlo)
            _merge_boundary(cur, lane, hhi)


def apply_batch(conn, lane, kind, batch_no, rows, uploaded_at):
    """应用一个批次(录入或整批更正)。调用方负责事务(BEGIN/COMMIT/ROLLBACK)。

    同一 (lane, kind, batch_no) 已存在时为整批更正: 旧行全部删除后写入新行,
    并只重算旧批和新批涉及的米段。
    """
    cur = conn.cursor()
    table = "repave_segments" if kind == "repave" else "inspect_segments"
    old = cur.execute(
        f"SELECT start_m,end_m FROM {table} WHERE lane=? AND batch_no=?",
        (lane, batch_no),
    ).fetchall()
    cur.execute(f"DELETE FROM {table} WHERE lane=? AND batch_no=?", (lane, batch_no))
    if kind == "repave":
        cur.executemany(
            "INSERT INTO repave_segments(lane,start_m,end_m,completed_at,batch_no) "
            "VALUES (?,?,?,?,?)",
            [(lane, s, e, t, batch_no) for s, e, t in rows],
        )
    else:
        cur.executemany(
            "INSERT INTO inspect_segments(lane,start_m,end_m,measured_at,device_id,run_id,batch_no) "
            "VALUES (?,?,?,?,?,?,?)",
            [(lane, s, e, t, d, r, batch_no) for s, e, t, d, r in rows],
        )
    cur.execute(
        "INSERT INTO batches(lane,kind,batch_no,uploaded_at,row_count) VALUES (?,?,?,?,?) "
        "ON CONFLICT(lane,kind,batch_no) DO UPDATE SET "
        "uploaded_at=excluded.uploaded_at, row_count=excluded.row_count",
        (lane, kind, batch_no, uploaded_at, len(rows)),
    )
    affected = [(r[0], r[1]) for r in old] + [(row[0], row[1]) for row in rows]
    recompute_components(conn, lane, merge_ranges(affected))


# ---------------------------------------------------------------- 查询

def query_acceptance(conn, lane, a, b, limit=5000):
    """读取已提交的交验结果(不做任何重算), 裁剪到 [a, b) 展示。"""
    rows = conn.execute(
        "SELECT * FROM acceptance WHERE lane=? AND start_m<? AND end_m>? "
        "ORDER BY start_m LIMIT ?",
        (lane, b, a, limit + 1),
    ).fetchall()
    truncated = len(rows) > limit
    out = []
    for r in rows[:limit]:
        d = dict(r)
        d["start_m"] = max(d["start_m"], a)
        d["end_m"] = min(d["end_m"], b)
        d["qualified"] = d["missing"] == 0
        out.append(d)
    return out, truncated
