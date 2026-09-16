# tracer.py — W14D2 산출물. 노드를 감싸는 포장지 하나.
# 노드 본문은 한 글자도 바꾸지 않는다. 그래서 미니 러너에도, LangGraph에도 그대로 붙는다.
import time

TRACE = []


def traced(name):
    def deco(fn):
        def wrapper(state):
            t0 = time.perf_counter()
            try:
                out = fn(state)
                status = "ok"
            except Exception as e:
                out, status = {}, "err:" + type(e).__name__
            TRACE.append({"node": name,
                          "ms": round((time.perf_counter() - t0) * 1000, 1),
                          "tokens": out.get("tokens", 0),
                          "status": status,
                          "doc": state.get("doc", "")})
            return out
        wrapper.__name__ = getattr(fn, "__name__", name)
        return wrapper
    return deco


def wrap_all(nodes):
    # 노드 딕셔너리를 통째로 포장한다.
    return {k: traced(k)(v) for k, v in nodes.items()}


def table(rows):
    # {node, ms, tokens, status} 기록을 노드별로 접어 병목을 지목한다.
    import pandas as pd
    df = pd.DataFrame(rows)
    g = df.groupby("node", as_index=False).agg(호출=("ms", "size"), 총ms=("ms", "sum"),
                                               평균ms=("ms", "mean"), 토큰=("tokens", "sum"))
    order = ["ingest", "classify", "compile", "gate", "store", "review"]
    g["_o"] = g["node"].map({n: i for i, n in enumerate(order)})
    g = g.sort_values("_o").drop(columns="_o")
    총 = g["총ms"].sum()
    # 키 없이 돌면 노드가 너무 빨라 총합이 0 이 된다. 0/0 은 nan 이라 화면에 그대로 찍힌다.
    g["비중%"] = (g["총ms"] / 총 * 100).round(1) if 총 > 0 else 0.0
    g["총ms"] = g["총ms"].round(1)
    g["평균ms"] = g["평균ms"].round(1)
    return g


def bottleneck(g):
    if g["총ms"].sum() == 0:                 # 규칙 경로는 너무 빨라 잴 것이 없다
        return "없음(키 없이 도는 경로 — 잴 만한 시간이 안 나옵니다)", 0.0, 0.0
    r = g.loc[g["총ms"].idxmax()]
    return r["node"], r["비중%"], r["평균ms"]
