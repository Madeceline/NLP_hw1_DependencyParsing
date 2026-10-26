'自行实现 Chu–Liu/Edmonds 算法；分数矩阵按“依存词、父节点”索引。'
import numpy as np


def _cycle(heads):
    done = {0}
    for start in range(1, len(heads)):
        node, path, positions = start, [], {}
        while node not in done:
            if node in positions:
                return path[positions[node]:]
            positions[node] = len(path)
            path.append(node)
            node = int(heads[node])
        done.update(path)
    return None


def _chu_liu(scores):
    '求解最大有向生成树；此阶段允许多个词直接依附人工根节点。'
    n = len(scores)
    heads = scores.argmax(axis=1)
    heads[0] = -1
    if not np.isfinite(scores[np.arange(1, n), heads[1:]]).all():
        raise ValueError("不存在可行的有向生成树")
    cycle = _cycle(heads)
    if cycle is None:
        return heads
    inside = np.array(cycle)
    cycle_set = set(cycle)
    outside = np.array([i for i in range(n) if i not in cycle_set])
    m = len(outside)
    contracted = np.full((m + 1, m + 1), -np.inf)
    contracted[:m, :m] = scores[np.ix_(outside, outside)]
    # 环外依存词的候选父节点可以位于环内。
    leaving = scores[np.ix_(outside, inside)]
    leaving_choice = leaving.argmax(axis=1)
    contracted[:m, m] = leaving[np.arange(m), leaving_choice]
    # 环外父节点接入环内时，需要断开一条已选中的环边。
    # 增量分数需减去被替换的边分数；环内总分是所有候选共有的常数。
    entering = scores[np.ix_(inside, outside)] - scores[inside, heads[inside]][:, None]
    entering_choice = entering.argmax(axis=0)
    contracted[m, :m] = entering[entering_choice, np.arange(m)]
    np.fill_diagonal(contracted, -np.inf)
    contracted[0, :] = -np.inf
    sub_heads = _chu_liu(contracted)
    result = heads.copy()
    for dep_index in range(1, m):
        head_index = sub_heads[dep_index]
        result[outside[dep_index]] = (inside[leaving_choice[dep_index]]
                                      if head_index == m else outside[head_index])
    entering_head = sub_heads[m]
    broken_dep = inside[entering_choice[entering_head]]
    result[broken_dep] = outside[entering_head]
    return result


def decode_mst(scores):
    '显式约束根词数量，精确求解得分最高的单根依存树。\n\n先求解允许多个根词的松弛问题；若结果不满足单根约束，\n则枚举所有可行的根词。逐行最大分数之和构成可采纳上界，\n用于安全剪枝，不改变最优解。'
    scores = np.asarray(scores, dtype=np.float64).copy()
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or len(scores) < 2:
        raise ValueError("分数矩阵必须为方阵，且包含 ROOT 和至少一个真实词")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("分数只能是有限值或负无穷")
    np.fill_diagonal(scores, -np.inf)
    scores[0, :] = -np.inf
    relaxed = _chu_liu(scores)
    if np.count_nonzero(relaxed[1:] == 0) == 1:
        return relaxed
    nonroot = scores[1:, 1:].max(axis=1)
    bounds = []
    for root in range(1, len(scores)):
        other = np.delete(nonroot, root - 1)
        bound = scores[root, 0] + other.sum()
        if np.isfinite(bound):
            bounds.append((float(bound), root))
    best, best_score = None, -np.inf
    for upper_bound, root in sorted(bounds, reverse=True):
        if upper_bound < best_score:
            break
        constrained = scores.copy()
        constrained[1:, 0] = -np.inf
        constrained[root, :] = -np.inf
        constrained[root, 0] = scores[root, 0]
        try:
            heads = _chu_liu(constrained)
        except ValueError:
            continue
        total = scores[np.arange(1, len(scores)), heads[1:]].sum()
        if total > best_score:
            best, best_score = heads, total
    if best is None:
        raise ValueError("不存在可行的单根依存树")
    return best


def validate_tree(heads):
    heads = np.asarray(heads)
    if len(heads) < 2 or heads[0] != -1:
        raise ValueError("人工根节点不能有父节点，其父节点编号应为 -1")
    if np.any(heads[1:] < 0) or np.any(heads[1:] >= len(heads)):
        raise ValueError("父节点索引超出范围")
    if np.any(heads[1:] == np.arange(1, len(heads))):
        raise ValueError("存在自环")
    if np.count_nonzero(heads[1:] == 0) != 1 or _cycle(heads) is not None:
        raise ValueError("依存树必须无环且只有一个根词")
    return True
