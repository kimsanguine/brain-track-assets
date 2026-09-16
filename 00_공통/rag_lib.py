"""rag_lib — 11~14주차 공용 검색·생성 도구상자.

핵심 원칙: **키 없이 끝까지 돈다.**
  임베딩 3단 폴백: LIVE(OpenRouter) → CACHE(.npz) → LOCAL(손구현 2-gram)
  생성 3단 폴백:   LIVE(OpenRouter) → CACHE(사전 응답) → RULE(규칙 생성기)
"""
from __future__ import annotations
import hashlib
import json
import os
import re

import numpy as np

OR_BASE = "https://openrouter.ai/api/v1"
EMB_MODEL = "openai/text-embedding-3-small"
CHAT_MODEL = "openai/gpt-4o-mini"
DIM_LOCAL = 512


# ══════════════════════════════════════════════════════════════════
# 1. 임베딩 — 3단 폴백
# ══════════════════════════════════════════════════════════════════
def _grams(text: str, n: int = 2) -> list[str]:
    t = re.sub(r"\s+", "", text)
    return [t[i:i + n] for i in range(max(0, len(t) - n + 1))] or [t]


def embed_local(texts: list[str], dim: int = DIM_LOCAL) -> np.ndarray:
    """손구현 임베딩: 2-gram을 해시로 칸에 넣고 길이를 1로 맞춘다.
    의미를 배우진 못하지만 '좌표로 바꾼다'는 원리는 같다."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for g in _grams(t):
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
            out[i, h % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def embed(texts: list[str], cache_path: str | None = None,
          model: str = EMB_MODEL, verbose: bool = True) -> tuple[np.ndarray, str]:
    """임베딩 3단 폴백. 반환: (벡터, 사용한 경로 이름)"""
    key = os.getenv("OPENROUTER_API_KEY")
    if key:                                                    # ① LIVE
        try:
            from openai import OpenAI
            cli = OpenAI(base_url=OR_BASE, api_key=key)
            r = cli.embeddings.create(model=model, input=texts)
            vecs = np.array([d.embedding for d in r.data], dtype=np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            if verbose:
                print(f"[임베딩] LIVE — {model} ({vecs.shape[1]}차원)")
            return vecs, "LIVE"
        except Exception as e:                                 # pragma: no cover
            if verbose:
                print(f"[임베딩] LIVE 실패({type(e).__name__}) → 캐시로 폴백")
    if cache_path and os.path.exists(cache_path):              # ② CACHE
        z = np.load(cache_path, allow_pickle=True)
        cached_texts = list(z["texts"])
        if cached_texts == list(texts):
            if verbose:
                print(f"[임베딩] CACHE — {os.path.basename(cache_path)} ({z['vecs'].shape[1]}차원)")
            return z["vecs"], "CACHE"
        if verbose:
            print("[임베딩] 캐시에 없는 문장 → 손구현으로 폴백")
    if verbose:                                                # ③ LOCAL
        print(f"[임베딩] LOCAL — 손구현 2-gram ({DIM_LOCAL}차원)")
    return embed_local(texts), "LOCAL"


def save_emb_cache(path: str, texts: list[str], vecs: np.ndarray) -> str:
    np.savez_compressed(path, texts=np.array(texts, dtype=object), vecs=vecs)
    return path


# ══════════════════════════════════════════════════════════════════
# 2. 검색 — 키워드 · 의미 · RRF 병합
# ══════════════════════════════════════════════════════════════════
def tokenize(text: str) -> list[str]:
    """단어 단위로 자른다(한글 조사는 간단히 떼어낸다)."""
    words = re.findall(r"[가-힣]+|[A-Za-z]+|\d[\d,]*", text)
    JOSA = ("으로", "에서", "에게", "까지", "부터", "이나", "라도", "는", "은", "이", "가",
            "을", "를", "의", "에", "도", "만", "과", "와", "로", "야", "요")
    out = []
    for w in words:
        for j in JOSA:
            if len(w) > len(j) + 1 and w.endswith(j):
                w = w[: -len(j)]
                break
        out.append(w)
    return out


def keyword_search(query: str, docs: list[str], topk: int = 5) -> list[tuple[int, float]]:
    """단어가 겹치는 정도로 찾는다(BM25 계열의 최소판 — 흔한 낱말은 가중치를 낮춘다).

    고유명사·숫자·조항 번호처럼 '그 낱말 그대로'가 중요한 질문에 강하다.
    """
    import math
    doc_tokens = [tokenize(d) for d in docs]
    N = len(docs)
    df: dict[str, int] = {}
    for toks in doc_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    scores = []
    for i, toks in enumerate(doc_tokens):
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for q in set(tokenize(query)):
            if q in tf:
                idf = math.log(1 + (N - df[q] + 0.5) / (df[q] + 0.5))
                s += idf * tf[q] / (tf[q] + 1.2 * (0.25 + 0.75 * len(toks) / 60))
        scores.append((i, round(s, 4)))
    return sorted(scores, key=lambda x: -x[1])[:topk]


def vector_search(query: str, docs: list[str], doc_vecs: np.ndarray,
                  cache_path: str | None = None, topk: int = 5) -> list[tuple[int, float]]:
    """의미가 가까운 정도로 찾는다(코사인 유사도)."""
    qv, _ = embed([query], cache_path=cache_path, verbose=False)
    sims = (doc_vecs @ qv[0])
    idx = np.argsort(-sims)[:topk]
    return [(int(i), float(sims[i])) for i in idx]


def rrf(rank_lists: list[list[tuple[int, float]]], k: int = 60, topk: int = 5):
    """Reciprocal Rank Fusion — 여러 검색 결과를 '순위'로 합친다.

    점수 체계가 달라도 순위는 비교할 수 있다는 것이 요점. 10줄이면 끝난다.
    """
    score: dict[int, float] = {}
    for lst in rank_lists:
        for rank, (doc_id, _) in enumerate(lst, start=1):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda x: -x[1])[:topk]


# ══════════════════════════════════════════════════════════════════
# 3. 생성 — 3단 폴백
# ══════════════════════════════════════════════════════════════════
def _rule_answer(question: str, contexts: list[str]) -> str:
    """규칙 생성기: 근거에서 질문 낱말이 가장 많이 겹치는 문장을 고른다."""
    if not contexts:
        return "근거를 찾지 못했습니다. 사람 확인이 필요합니다."
    qg = set(_grams(question))
    best, best_s = contexts[0], -1.0
    for c in contexts:
        for sent in re.split(r"(?<=[.。])\s*|\n", c):
            if not sent.strip():
                continue
            s = len(qg & set(_grams(sent))) / (len(set(_grams(sent))) ** 0.5 + 1e-9)
            if s > best_s:
                best, best_s = sent.strip(), s
    return best


def generate(question: str, contexts: list[str], cache: dict | None = None,
             model: str = CHAT_MODEL, system: str | None = None,
             verbose: bool = True) -> tuple[str, str]:
    """생성 3단 폴백. 반환: (답변, 사용한 경로 이름)"""
    key = os.getenv("OPENROUTER_API_KEY")
    if key:                                                    # ① LIVE
        try:
            from openai import OpenAI
            cli = OpenAI(base_url=OR_BASE, api_key=key)
            ctx = "\n\n".join(f"[근거 {i+1}]\n{c}" for i, c in enumerate(contexts))
            msgs = [{"role": "system", "content": system or
                     "너는 카드사 상담 지원 AI다. 반드시 주어진 [근거] 안에서만 답하고, "
                     "근거에 없으면 '근거 없음'이라고 말한다."},
                    {"role": "user", "content": f"{ctx}\n\n질문: {question}"}]
            r = cli.chat.completions.create(model=model, messages=msgs, temperature=0)
            if verbose:
                print(f"[생성] LIVE — {model}")
            return r.choices[0].message.content.strip(), "LIVE"
        except Exception as e:                                 # pragma: no cover
            if verbose:
                print(f"[생성] LIVE 실패({type(e).__name__}) → 캐시로 폴백")
    if cache and question in cache:                            # ② CACHE
        if verbose:
            print("[생성] CACHE — 사전 준비된 응답")
        return cache[question], "CACHE"
    if verbose:                                                # ③ RULE
        print("[생성] RULE — 규칙 생성기(근거에서 문장 선택)")
    return _rule_answer(question, contexts), "RULE"


# ══════════════════════════════════════════════════════════════════
# 4. 문서 청킹 · 로딩
# ══════════════════════════════════════════════════════════════════
def load_terms_chunks(pdf_path: str) -> list[dict]:
    """약관 PDF를 조항 단위로 자른다. 반환: [{id, title, text}]"""
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        raw = "\n".join((p.extract_text() or "") for p in pdf.pages)
    parts = re.split(r"(제\d+조(?:의\d+)?\([^)]+\))", raw)
    chunks = []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = re.sub(r"\s+", " ", parts[i + 1]).strip() if i + 1 < len(parts) else ""
        chunks.append({"id": title.split("(")[0], "title": title, "text": f"{title} {body}"})
    return chunks
