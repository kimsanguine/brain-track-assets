"""tools.py — 내가 만든 도구 2개 (W13D1).

도구 3요소 = 이름(name) · 설명(description) · 입력 스키마(parameters)
AI는 함수 본문을 못 본다. **이름과 설명만 보고 고른다.**
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "00_공통")))
import rag_lib as R

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))
LIMIT_DB = {
    "C-1024": {"이용한도": 5_000_000, "이번달사용": 1_384_200},
    "C-2048": {"이용한도": 2_000_000, "이번달사용": 1_950_000},
}
_CACHE = []


def terms_chunks():
    global _CACHE
    if not _CACHE:
        _CACHE = R.load_terms_chunks(os.path.join(ASSETS, "terms.pdf"))
    return _CACHE


def search_card_terms(query: str, top_k: int = 2) -> dict:
    """약관에서 조항을 찾아 돌려준다(더미 = 로컬 약관 PDF)."""
    ch = terms_chunks()
    hits = R.keyword_search(query, [c["text"] for c in ch], topk=top_k)
    return {"query": query,
            "hits": [{"조항": ch[i]["id"], "제목": ch[i]["title"], "본문": ch[i]["text"]}
                     for i, s in hits if s > 0]}


def get_card_limit(member_id: str) -> dict:
    """회원번호로 한도를 조회한다(더미 DB)."""
    if member_id not in LIMIT_DB:
        raise KeyError(f"회원번호 없음: {member_id}")
    r = LIMIT_DB[member_id]
    return {"회원번호": member_id, "이용한도": r["이용한도"], "이번달사용": r["이번달사용"],
            "잔여한도": r["이용한도"] - r["이번달사용"]}


GOOD = [
    {"name": "search_card_terms",
     "description": ("카드 약관 조항 검색기. 연회비 · 면제 · 할부 · 포인트 · 분실 · 도난 · 해지 "
                     "규정을 찾을 때 쓴다. 조항 번호와 본문을 돌려준다."),
     "parameters": {"type": "object",
                    "properties": {"query": {"type": "string", "description": "찾을 말"},
                                   "top_k": {"type": "integer", "default": 2}},
                    "required": ["query"]},
     "fn": search_card_terms},
    {"name": "get_card_limit",
     "description": ("카드 한도 조회기. 회원번호로 이용 한도와 잔여 한도를 원 단위로 알려준다. "
                     "'한도가 얼마인지', '얼마나 더 쓸 수 있는지'를 묻는 질문에 쓴다."),
     "parameters": {"type": "object",
                    "properties": {"member_id": {"type": "string",
                                                 "description": "회원번호(예: C-1024)"}},
                    "required": ["member_id"]},
     "fn": get_card_limit},
]

BAD = [
    {"name": "func_a", "description": "내부 함수 A. 데이터를 처리한다.",
     "parameters": GOOD[0]["parameters"], "fn": search_card_terms},
    {"name": "func_b", "description": "내부 함수 B. 값을 반환한다.",
     "parameters": GOOD[1]["parameters"], "fn": get_card_limit},
]
