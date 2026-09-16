"""w11_embed_search.py — 내 위키를 의미로 찾는다 (이식 ②).

사용:
    python3 extensions/w11_embed_search.py --build        # 위키를 좌표로 바꿔 저장
    python3 extensions/w11_embed_search.py "지난달 그 회고"  # 의미로 찾기
    (윈도우는 python3 자리에 py 를 넣으세요: py extensions/...)

키워드 검색은 **쓴 단어 그대로** 쳐야 찾힌다. 흐릿하게 기억나는 것은 그렇게 못 찾는다.
글을 좌표(벡터)로 바꿔 두면 "비슷한 자리에 있는 것"을 찾을 수 있다.

임베딩은 3단으로 떨어진다. 어느 경로로 돌았는지는 항상 화면에 찍힌다.

    LIVE  — OPENROUTER_API_KEY 가 있을 때. 진짜 의미를 배운 모델이 좌표를 만든다.
    CACHE — 전에 만들어 둔 .npz 가 있을 때. 같은 문서는 다시 부르지 않는다(=돈을 안 쓴다).
    LOCAL — 둘 다 없을 때. 2-gram 을 해시로 칸에 넣는 손구현.
            의미를 배우진 못하지만 "좌표로 바꾼다"는 원리는 같고, **키 없이 완주한다.**

증분: 어떤 파일을 이미 넣었는지 .npz 에 함께 적어 두고 비교한다.
새 문서만 임베딩하면 되고, 이건 다음 날 배울 **멱등성**과 같은 발상이다.

접점은 하나뿐. `wiki/` 읽기.
"""
import hashlib
import os
import re
import sys

import numpy as np

BRAIN_HOME = os.environ.get("BRAIN_HOME") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
WIKI = os.path.join(BRAIN_HOME, "wiki")
CACHE = os.path.join(BRAIN_HOME, ".embed_cache.npz")
EMB_MODEL = "openai/text-embedding-3-small"
OR_BASE = "https://openrouter.ai/api/v1"
DIM_LOCAL = 512


# ---------------------------------------------------------------------------
# 위키 읽기
# ---------------------------------------------------------------------------


def _fingerprint(text: str) -> str:
    """문서 내용의 지문. slug 만 비교하면 본문을 고쳐도 옛 벡터를 그대로 쓴다."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def load_wiki() -> tuple[list, list]:
    """wiki/ 의 md 를 (slug, 본문) 으로 읽는다. frontmatter 는 걷어낸다."""
    slugs, texts = [], []
    if not os.path.isdir(WIKI):
        return slugs, texts
    for root, _dirs, files in os.walk(WIKI):
        for fn in sorted(files):
            if not fn.endswith(".md"):
                continue
            with open(os.path.join(root, fn), encoding="utf-8", errors="replace") as f:
                raw = f.read()
            body = re.sub(r"^---.*?---", "", raw, count=1, flags=re.S).strip()
            slugs.append(os.path.splitext(fn)[0])
            texts.append(" ".join(body.split())[:2000])
    return slugs, texts


# ---------------------------------------------------------------------------
# 임베딩 3단 폴백
# ---------------------------------------------------------------------------


def _grams(text: str, n: int = 2) -> list:
    t = re.sub(r"\s+", " ", text)
    return [t[i:i + n] for i in range(max(0, len(t) - n + 1))]


def embed_local(texts: list, dim: int = DIM_LOCAL) -> np.ndarray:
    """손구현 임베딩: 2-gram 을 해시로 칸에 넣고 길이를 1로 맞춘다.

    의미를 배우진 못한다. 다만 "글을 좌표로 바꾼다"는 원리는 같고, 키가 없어도 돈다.
    """
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for g in _grams(t):
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
            out[i, h % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def embed_live(texts: list) -> np.ndarray:
    """OpenRouter 임베딩 API. 키가 없거나 실패하면 예외를 올린다(조용히 넘기지 않는다)."""
    from openai import OpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY 없음")
    cli = OpenAI(base_url=OR_BASE, api_key=key)
    r = cli.embeddings.create(model=EMB_MODEL, input=texts)
    vecs = np.array([d.embedding for d in r.data], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def embed(texts: list, verbose: bool = True) -> tuple:
    """임베딩 3단 폴백. 반환: (벡터, 사용한 경로 이름)"""
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            vecs = embed_live(texts)
            if verbose:
                print(f"[임베딩] LIVE — {EMB_MODEL} ({vecs.shape[1]}차원)")
            return vecs, "LIVE"
        except Exception as exc:
            if verbose:
                print(f"[임베딩] LIVE 실패({type(exc).__name__}) → LOCAL 로 내려갑니다")
    vecs = embed_local(texts)
    if verbose:
        print(f"[임베딩] LOCAL — 2-gram 해시 ({vecs.shape[1]}차원, 키 없이 동작)")
    return vecs, "LOCAL"


# ---------------------------------------------------------------------------
# 인덱스 (증분)
# ---------------------------------------------------------------------------


def build_index(verbose: bool = True) -> dict:
    """위키를 좌표로 바꿔 저장한다. **이미 넣은 문서는 다시 부르지 않는다.**"""
    slugs, texts = load_wiki()
    if not slugs:
        print("  위키가 비어 있습니다. 먼저 python3 scripts/compile.py 를 실행하세요.")
        return {"total": 0, "new": 0, "path": "-"}

    # 캐시 열쇠는 slug 가 아니라 "slug + 내용 지문"이다.
    # slug 만 보면 위키를 고쳐도 다시 임베딩하지 않아, 검색이 옛 내용을 가리킨다.
    keys = [f"{sl}#{_fingerprint(tx)}" for sl, tx in zip(slugs, texts)]

    old_keys, old_vecs = [], None
    if os.path.exists(CACHE):
        try:
            z = np.load(CACHE, allow_pickle=True)
            old_keys, old_vecs = list(z["keys"]), z["vecs"]
        except (KeyError, ValueError, OSError):
            old_keys, old_vecs = [], None      # 옛 형식 캐시 → 통째로 다시 만든다

    known = {k: i for i, k in enumerate(old_keys)}
    todo = [i for i, k in enumerate(keys) if k not in known]

    if not todo and old_vecs is not None and len(old_keys) == len(keys):
        if verbose:
            print(f"[임베딩] CACHE — 새 문서 0건, {len(slugs)}편 그대로 씁니다")
        return {"total": len(slugs), "new": 0, "path": "CACHE"}

    new_vecs, path = embed([texts[i] for i in todo], verbose=verbose) if todo else (None, "CACHE")

    dim = new_vecs.shape[1] if new_vecs is not None else old_vecs.shape[1]
    vecs = np.zeros((len(slugs), dim), dtype=np.float32)
    k = 0
    for i, key in enumerate(keys):
        if key in known and old_vecs is not None and old_vecs.shape[1] == dim:
            vecs[i] = old_vecs[known[key]]
        else:
            vecs[i] = new_vecs[k]
            k += 1

    np.savez(CACHE, slugs=np.array(slugs), keys=np.array(keys), vecs=vecs)
    if verbose:
        print(f"  {len(slugs)}편 저장 (새로 임베딩 {len(todo)}편) → {os.path.basename(CACHE)}")
    return {"total": len(slugs), "new": len(todo), "path": path}


def search(query: str, top_k: int = 3, verbose: bool = True) -> list:
    """질문을 같은 좌표계로 옮겨 가장 가까운 문서를 찾는다."""
    if not os.path.exists(CACHE):
        build_index(verbose=verbose)
    if not os.path.exists(CACHE):
        return []

    z = np.load(CACHE, allow_pickle=True)
    slugs, vecs = list(z["slugs"]), z["vecs"]

    # 질문도 문서와 같은 방식으로 좌표를 만들어야 비교가 성립한다
    qv, _ = embed([query], verbose=False)
    if qv.shape[1] != vecs.shape[1]:
        qv = embed_local([query], dim=vecs.shape[1])

    sims = (vecs @ qv[0]).astype(float)
    order = np.argsort(-sims)[:top_k]
    return [{"slug": slugs[i], "score": round(float(sims[i]), 4)} for i in order]


def main(argv) -> int:
    if not argv or "--build" in argv:
        r = build_index()
        print(f"[embed_search] 문서 {r['total']}편 · 새로 임베딩 {r['new']}편 · 경로 {r['path']}")
        if not argv:
            print('\n찾아보기: python3 extensions/w11_embed_search.py "지난달 그 회고"')
        return 0

    q = argv[0]
    rows = search(q)
    print(f'[embed_search] "{q}" → {len(rows)}건')
    for r in rows:
        print(f"  · {r['slug']}  (유사도 {r['score']})")
    if not rows:
        print("  위키가 비어 있습니다. python3 scripts/compile.py 를 먼저 실행하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
