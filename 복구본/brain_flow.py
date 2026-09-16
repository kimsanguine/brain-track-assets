# brain_flow.py — W14D1 산출물. 내 브레인의 흐름 그래프.
# 위 셀들에서 만든 것을 그대로 파일로 굳힌 것이다(D2·D4가 import 한다).
import os, sys, time
from typing import TypedDict
sys.path.insert(0, os.path.abspath("../../00_공통"))
import rag_lib as R

try:
    from langgraph.graph import StateGraph, START, END
    HAS_LG = True
except ImportError:
    HAS_LG = False
    START, END = "START", "END"

MODELS = {"openai/gpt-4o-mini": {"in": 0.15, "out": 0.60, "mock_ms": 25, "급": "저가"},
          "openai/gpt-4o":      {"in": 2.50, "out": 10.00, "mock_ms": 90, "급": "상위"}}
CLS_MODEL, GEN_MODEL, THRESHOLD = "openai/gpt-4o-mini", "openai/gpt-4o", 0.7


class BrainState(TypedDict, total=False):
    doc: str          # 파일명
    text: str         # 원문
    doc_type: str     # statement / terms / notice / note
    note: str         # 컴파일된 위키 노트
    quality: float    # 게이트 점수 0~1
    dest: str         # wiki / review   ← 오늘의 주인공 칸
    path: list        # 방문 노드 기록
    tokens: int       # 그 노드가 쓴 토큰 수


def call_llm(question, contexts, model, system=None):
    # 키가 있으면 LIVE, 없으면 rag_lib의 RULE 폴백.
    # 폴백에는 네트워크 지연이 없어 ms가 0으로 찍힌다 → 폴백일 때만 모의 지연을 넣어
    # 운영의 모양을 재현한다(숫자는 모의값, 재는 법이 학습 대상).
    out, route = R.generate(question, contexts, model=model, system=system, verbose=False)
    if route != "LIVE":
        time.sleep(MODELS[model]["mock_ms"] / 1000)
    tok_in = (len(question) + sum(len(c) for c in contexts)) // 2
    return out, tok_in + max(len(out) // 2, 1)


LABELS = ("statement", "terms", "notice", "note")


def rule_classify(t):
    if "명세서" in t:
        return "statement"
    if "약관" in t:
        return "terms"
    if "공지" in t:
        return "notice"
    return "note"


def ingest_node(state):
    return {"text": state["text"].strip(), "path": state.get("path", []) + ["ingest"]}


def classify_node(state):
    out, tok = call_llm("이 문서의 유형은 statement, terms, notice, note 중 무엇인가? 한 단어로만 답하라.",
                        [state["text"]], CLS_MODEL,
                        system="너는 문서 분류기다. statement/terms/notice/note 중 한 단어만 출력한다.")
    dt = out.strip().lower() if out.strip().lower() in LABELS else rule_classify(state["text"])
    return {"doc_type": dt, "path": state["path"] + ["classify"], "tokens": tok}


def compile_node(state):
    out, tok = call_llm("이 문서의 핵심을 한 줄로 요약하라.", [state["text"]], GEN_MODEL)
    first = out.split(".")[0][:40] if out else state["text"][:40]
    return {"note": "[" + state["doc_type"] + "] " + first,
            "path": state["path"] + ["compile"], "tokens": tok}


def gate_v1(state):
    # v1 — 길이 x 한글 비율. 숫자를 정보로 세지 않는다.
    t = state["text"]
    info = sum(1 for c in t if "가" <= c <= "힣")
    return {"quality": round(min(len(t) / 30, 1.0) * (info / max(len(t.replace(" ", "")), 1)), 2),
            "path": state["path"] + ["gate"]}


def gate_v2(state):
    # v2 — 한 줄만 고친다: 숫자도 정보로 인정한다.
    t = state["text"]
    info = sum(1 for c in t if ("가" <= c <= "힣") or c.isdigit())
    return {"quality": round(min(len(t) / 30, 1.0) * (info / max(len(t.replace(" ", "")), 1)), 2),
            "path": state["path"] + ["gate"]}


def store_node(state):
    return {"dest": "wiki", "path": state["path"] + ["store"]}


def review_node(state):
    return {"dest": "review", "path": state["path"] + ["review"]}


def route_fn(state):
    # 분기 함수 — 반환값은 반드시 노드 이름이다.
    return "store" if state["quality"] >= THRESHOLD else "review"


def make_nodes(gate=gate_v2):
    return {"ingest": ingest_node, "classify": classify_node, "compile": compile_node,
            "gate": gate, "store": store_node, "review": review_node}


EDGES = {"START": "ingest", "ingest": "classify", "classify": "compile",
         "compile": "gate", "gate": route_fn, "store": "END", "review": "END"}


def run(state, nodes, edges):
    # 미니 러너 — 열 줄 남짓. 노드는 다음을 모른다.
    cur = "START"
    while True:
        nxt = edges[cur]
        cur = nxt(state) if callable(nxt) else nxt
        if cur == "END":
            return state
        state = {**state, **nodes[cur](state)}


def build_graph(nodes):
    # 같은 노드, 실행기만 교체. langgraph가 없으면 None을 돌려준다.
    if not HAS_LG:
        return None
    g = StateGraph(BrainState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "classify")
    g.add_edge("classify", "compile")
    g.add_edge("compile", "gate")
    g.add_conditional_edges("gate", route_fn, {"store": "store", "review": "review"})
    g.add_edge("store", END)
    g.add_edge("review", END)
    return g.compile()


SEED_DOCS = [
    {"doc": "card_statement_01.md", "text": "9월 이용대금 명세서. 결제일 2026-09-25, 청구금액 384,200원. 할부 잔여 2회."},
    {"doc": "terms_annual_fee.md",  "text": "약관 제12조(연회비). 연회비는 카드 등급에 따라 부과하며 최초 연도 면제 조건은 별표3을 따른다."},
    {"doc": "meeting_note_0902.md", "text": "회의 메모: 10기 브레인 트랙 — 매주 extensions/에 장기 하나씩. cron은 11주차."},
    {"doc": "notice_event.md",      "text": "공지: 9월 무이자 할부 이벤트. 대상 가맹점은 첨부 목록 참조."},
    {"doc": "card_statement_02.md", "text": "8월 이용대금 명세서. 결제일 2026-08-25, 청구금액 291,700원."},
    {"doc": "scrap_blank.md",       "text": "ㅁ"},
    {"doc": "terms_limit.md",       "text": "약관 제7조(이용한도). 한도는 심사 기준에 따라 조정될 수 있다."},
    {"doc": "memo_idea.md",         "text": "아이디어 메모: 검수함에 쌓인 문서를 아침에 한 번에 보는 화면."},
    {"doc": "ocr_noise.md",         "text": "€∆ㅁ2ㅐ.. ㅣ|l1 O0"},
    {"doc": "notice_change.md",     "text": "공지: 10월부터 명세서 발송이 이메일 기본으로 변경됩니다."},
]
