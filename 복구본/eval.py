"""eval.py — W12D3 산출물. 골든 문항 규칙 채점 3종."""
import json, os, re, sys
sys.path.insert(0, os.path.abspath("../../00_공통"))
import rag_lib as R

MONEY = re.compile(r"\d{1,3}(?:,\d{3})+")
BANNED = ["1년 면제", "첫 해 면제", "첫해 면제", "1년간 무료", "연회비 없음",
          "근거를 찾지 못했습니다", "확인 불가"]


def r_source(card, g):
    """규칙1 출처 일치 — 카드의 근거 목록에 정답 출처가 있는가"""
    return g["source"] in card["근거"]


def r_amount(card, g):
    """규칙2 금액 일치 — 정답 금액이 결론에 있고 다른 금액이 섞이지 않았는가
    (정답에 금액이 없으면 정답 낱말이 결론에 모두 들어 있는가)"""
    want, got = set(MONEY.findall(g["answer"])), set(MONEY.findall(card["결론"]))
    if want:
        return want <= got and not (got - want)
    return all(t in card["결론"] for t in R.tokenize(g["answer"]))


def r_banned(card, g):
    """규칙3 금지어 — 흔한 오답 표현이 결론에 들어 있지 않은가"""
    return not any(b in card["결론"] for b in BANNED)


RULES = [("출처", r_source), ("금액", r_amount), ("금지어", r_banned)]


def load_golden(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def grade(cards, golden, label, detail=False):
    """카드 목록을 채점해 (통과 수, 상세 행)을 돌려준다"""
    rows, npass = [], 0
    for g, card in zip(golden, cards):
        flags = [f(card, g) for _, f in RULES]
        ok = all(flags)
        npass += ok
        rows.append((g["id"], ok, flags, card))
        if detail:
            mark = "".join("O" if f else "X" for f in flags)
            print(f"  {g['id']} {'PASS' if ok else 'FAIL'} [{mark}] "
                  f"근거={','.join(card['근거']):12} {card['결론'][:34]}")
    print(f"  -> {label}: {npass}/{len(golden)} 통과")
    return npass, rows
