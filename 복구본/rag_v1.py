"""rag_v1.py — W12D1 산출물. RAG 4단(검색→조립→생성→검증) + 3칸 카드 계약."""
import os, re, sys
sys.path.insert(0, os.path.abspath("../../00_공통"))
import rag_lib as R

SENT   = re.compile(r"(?<=[.。])\s*|\n")
MONEY  = re.compile(r"\d{1,3}(?:,\d{3})+")
CAVEAT = ("단,", "다만,", "않으며", "이전에 발급", "완제 후", "변동 시")
PDF    = os.path.join(os.path.abspath("../../assets"), "terms.pdf")


def chunk(key, cid, ctx, body):
    """청크 하나. text=검색용(문맥 요약 포함) · body=생성용(사실 문장만)"""
    return {"key": key, "id": cid, "ctx": ctx, "body": body,
            "text": (f"[문맥] {ctx}\n{body}" if ctx else body)}


def corpus_v0():
    """PDF를 조항 단위로 자른 것 그대로 — 오늘의 출발점"""
    return [chunk(c["id"], c["id"], "", c["text"]) for c in R.load_terms_chunks(PDF)]


def search(q, corpus, vecs=None, mode="keyword", topk=3):
    """① 검색 — keyword 단독 또는 keyword+의미 RRF 병합(11주차 자산)"""
    docs = [c["text"] for c in corpus]
    kw = R.keyword_search(q, docs, topk=5)
    if mode == "keyword":
        return kw[:topk]
    ve = R.vector_search(q, docs, vecs, topk=5)
    return R.rrf([kw, ve], topk=topk)


def assemble(hits, corpus):
    """② 조립 — 출처 태그를 붙여 순서대로 쌓는다.
    문맥 요약(ctx)은 **검색용**이므로 생성에는 본문(body)만 넣는다."""
    return [f"[출처 {corpus[i]['id']}] {corpus[i]['body']}" for i, _ in hits]


def clean(a):
    return re.sub(r"^\[[^\]]*\]\s*", "", a.strip()).strip()


def sents(t):
    return [s.strip() for s in SENT.split(t) if s.strip()]


def verify(hits, corpus, prim_key, concl):
    """④ 검증 — 확인 칸을 채운다.
    ⓐ 적용 범위: 근거 청크의 문맥 요약(언제·누구에게 적용되는 근거인가)
    ⓑ 단서: 같은 조항의 다른 청크에 붙은 예외 문장"""
    prim = next(c for c in corpus if c["key"] == prim_key)
    parts = [prim["ctx"]] if prim["ctx"] else []
    for i, _ in hits:
        c = corpus[i]
        if c["id"] != prim["id"]:
            continue
        for s in sents(c["body"]):
            if s != concl and any(m in s for m in CAVEAT):
                parts.append(s)
                break
    return " / ".join(parts) if parts else "추가 확인사항 없음"


def answer_card(q, corpus, vecs=None, mode="keyword", with_check=True, topk=3):
    """RAG 4단을 한 번에 돌려 3칸 카드 {결론, 근거, 확인}를 만든다"""
    hits = search(q, corpus, vecs, mode, topk)                     # ①
    ctxs = assemble(hits, corpus)                                  # ②
    concl, path = R.generate(q, ctxs, verbose=False)               # ③
    concl = clean(concl)
    prim_key = next((corpus[i]["key"] for i, _ in hits if concl[:14] in corpus[i]["body"]),
                    corpus[hits[0][0]]["key"])
    prim = next(c for c in corpus if c["key"] == prim_key)
    chk = verify(hits, corpus, prim_key, concl) if with_check else "—"
    return {"결론": concl, "근거": [prim["id"]], "확인": chk, "경로": path}


def show(card, w=62):
    print(f"  결론: {card['결론'][:w]}")
    print(f"  근거: {' · '.join(card['근거'])}")
    print(f"  확인: {card['확인'][:w]}")
