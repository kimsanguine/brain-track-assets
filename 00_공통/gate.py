"""게이트 검사 — 실행된 노트북에서 에러/경고를 잡고, 교안 대조용 출력을 뽑는다.
사용: python3 gate.py <노트북 경로 또는 폴더> [--print]
"""
import json, os, sys, glob


def check(path, show=False):
    nb = json.load(open(path, encoding="utf-8"))
    errs, warns, outs = [], [], []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errs.append(f"셀{i}: {o.get('ename')}: {''.join(o.get('evalue', ''))[:120]}")
            if o.get("output_type") == "stream" and o.get("name") == "stderr":
                warns.append(f"셀{i}: {''.join(o.get('text', ''))[:160]}")
            if o.get("output_type") == "stream" and o.get("name") == "stdout":
                outs.append("".join(o.get("text", "")))
            if o.get("output_type") == "execute_result":
                outs.append("".join(o["data"].get("text/plain", "")))
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    n_run = sum(1 for c in nb["cells"] if c["cell_type"] == "code" and c.get("execution_count"))
    status = "PASS" if not errs and not warns else ("FAIL" if errs else "WARN")
    print(f"[{status}] {os.path.basename(path):34} 셀 {n_run}/{n_code} 실행 · 에러 {len(errs)} · stderr {len(warns)}")
    for e in errs:
        print("   ❌", e)
    for w in warns[:3]:
        print("   ⚠️", w.replace("\n", " ")[:140])
    if show:
        print("\n".join(outs))
    return not errs and not warns


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    show = "--print" in sys.argv
    files = [target] if target.endswith(".ipynb") else sorted(glob.glob(os.path.join(target, "**", "*.ipynb"), recursive=True))
    ok = all(check(f, show) for f in files)
    print(("\n전체 통과" if ok else "\n실패 항목 있음") + f" — {len(files)}개 노트북")
    sys.exit(0 if ok else 1)
