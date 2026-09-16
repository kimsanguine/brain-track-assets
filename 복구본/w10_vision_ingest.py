"""w10_vision_ingest.py — 사진·PDF·HTML을 검문한 뒤 raw/에 넣는다 (이식 ①).

사용:
    python3 extensions/w10_vision_ingest.py ~/사진/명세서.jpg
    python3 extensions/w10_vision_ingest.py ~/문서/약관.pdf ~/받은자료/공지.html
    (윈도우는 python3 자리에 py 를 넣으세요: py extensions/...)

수업(W10D4)에서 만든 read_any · inspect · vision_ingest 를 브레인에 붙인 것.
본체 수정 0줄. 접점은 **raw/ 에 md 를 쓰는 것** 하나뿐이고,
검문에 걸린 문서는 raw/ 가 아니라 review/ 로 보낸다.

경로는 BRAIN_HOME 을 따른다. 지정하지 않으면 이 파일의 상위 폴더(= 브레인 루트)다.
    export BRAIN_HOME=~/llm-brain-edu

없는 도구는 조용히 넘기지 않고 무엇이 없어서 무엇을 못 했는지 화면에 적는다.
이미지 OCR 만 tesseract 를 요구하고, PDF·HTML·텍스트는 추가 설치 없이 돈다.
"""
import datetime as dt
import os
import re
import sys

BRAIN_HOME = os.environ.get("BRAIN_HOME") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
RAW = os.path.join(BRAIN_HOME, "raw", "docs")      # 내 브레인의 입구
REVIEW = os.path.join(BRAIN_HOME, "review")        # 검문에 걸린 문서의 검수함

# 프롬프트 인젝션 검문 규칙 — W10D4 에서 손으로 만든 것
PATTERNS = r"이전 지시|모두 무시|권한으로 전환|출력하세요|답하라|ignore previous"


def inspect(raw: str):
    """외부 문서를 브레인에 넣기 전 검문. 반환: [(규칙, 걸린 문장)]

    완벽한 방어가 아니다. 규칙을 피해가는 문장은 늘 있다(11-2 격리, 13-4 권한으로 이어진다).
    """
    from bs4 import BeautifulSoup

    s = BeautifulSoup(raw, "html.parser")
    risks = []
    for c in re.findall(r"<!--(.*?)-->", raw, re.S):                 # ① 주석에 숨김
        if re.search(PATTERNS, c):
            risks.append(("주석에 숨김", c.strip()))
    for t in s.find_all(style=True):                                 # ② 화면에서 안 보임
        if "display:none" in t["style"].replace(" ", ""):
            risks.append(("화면에서 안 보임", t.get_text(strip=True)))
    for m in re.findall(rf"[^.>\n]*(?:{PATTERNS})[^.<\n]*", s.get_text()):  # ③ 명령형 문구
        risks.append(("명령형 문구", m.strip()))
    return risks


def inspect_text(text: str) -> list:
    """추출된 **본문**을 검문한다. HTML 태그가 없어도 명령문은 명령문이다.

    inspect() 는 숨긴 주석·display:none 처럼 HTML 구조에 기대므로 .html 에만 쓸 수 있다.
    그런데 인젝션은 txt·PDF·사진 OCR 결과로도 들어온다 — 그 경로가 열려 있으면
    "검문하고 들인다"는 규칙이 절반만 지켜진다.
    """
    risks = []
    for line in text.splitlines():
        line = line.strip()
        if line and re.search(PATTERNS, line):
            risks.append(("본문 명령형 문구", line[:120]))
    return risks


def read_any(path: str) -> str:
    """사진·PDF·HTML·텍스트를 글자로 바꾼다.

    수업 노트북은 pdfplumber 와 수업용 ocr_lib 를 썼지만, 브레인에는 이미 pymupdf 가
    들어 있으므로 그것을 쓴다(추가 설치 0). 이미지 OCR 만 tesseract 가 필요하다.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise RuntimeError(
                "이미지를 읽으려면 tesseract 가 필요합니다.\n"
                "  맥: brew install tesseract tesseract-lang && python3 -m pip install pytesseract\n"
                "  윈도우: https://github.com/UB-Mannheim/tesseract 설치 후 python3 -m pip install pytesseract\n"
                "  (PDF·HTML·텍스트는 지금도 그냥 됩니다)"
            )
        try:
            return pytesseract.image_to_string(Image.open(path), lang="kor+eng")
        except Exception as exc:
            # pytesseract 는 설치됐지만 tesseract 실행 파일이 없는 흔한 경우.
            # 입문자에게 traceback 을 던지지 않고 다음에 무엇을 할지 알려 준다.
            raise RuntimeError(
                f"이미지를 읽지 못했습니다({type(exc).__name__}).\n"
                "  tesseract 프로그램이 없거나 한국어 데이터가 빠졌을 수 있습니다.\n"
                "  맥: brew install tesseract tesseract-lang\n"
                "  윈도우: https://github.com/UB-Mannheim/tesseract 에서 설치\n"
                "  (PDF·HTML·텍스트는 지금도 그냥 됩니다)"
            ) from exc

    if ext == ".pdf":
        # 브레인에는 pymupdf 가, 수업 노트북 환경에는 pdfplumber 가 있다.
        # 어느 쪽에 두어도 돌아야 하므로 있는 것을 쓴다.
        try:
            import fitz  # pymupdf

            with fitz.open(path) as doc:
                return "\n".join(page.get_text() for page in doc)
        except ImportError:
            pass
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                return "\n".join((p.extract_text() or "") for p in pdf.pages)
        except ImportError:
            raise RuntimeError(
                "PDF 를 읽으려면 pymupdf 나 pdfplumber 중 하나가 필요합니다.\n"
                "  python3 -m pip install pymupdf     (브레인 기본 의존성)\n"
                "  (이미지·HTML·텍스트는 지금도 그냥 됩니다)"
            )

    if ext in (".html", ".htm"):
        from bs4 import BeautifulSoup

        with open(path, encoding="utf-8", errors="replace") as f:
            return BeautifulSoup(f.read(), "html.parser").get_text(" ", strip=True)

    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def vision_ingest(path: str) -> dict:
    """읽기 → 검문 → (통과분만) raw/ 저장. 걸리면 review/ 로 보낸다."""
    for d in (RAW, REVIEW):
        os.makedirs(d, exist_ok=True)

    # 검문은 HTML 원문(숨긴 주석·display:none)을 봐야 하므로 텍스트 추출 전에 읽는다
    raw_html = ""
    if path.lower().endswith((".html", ".htm")):
        with open(path, encoding="utf-8", errors="replace") as f:
            raw_html = f.read()
    risks = inspect(raw_html) if raw_html else []

    text = read_any(path)
    # HTML 이 아니어도 본문은 검문한다(txt·PDF·OCR 결과에도 명령문이 들어온다).
    # HTML 은 위에서 이미 봤으므로 중복 적발만 피한다.
    if not raw_html:
        risks += inspect_text(text)
    name = os.path.splitext(os.path.basename(path))[0]
    dest = REVIEW if risks else RAW
    verdict = "위험 %d건 → 검수함" % len(risks) if risks else "통과"

    note = [
        "---",
        f"title: {name}",
        f"created: {dt.datetime.now():%Y-%m-%d}",
        f"source: {os.path.basename(path)}",
        "---",
        "",
        f"# {name}",
        "",
        f"- 출처: {os.path.basename(path)}",
        f"- 반입 시각: {dt.datetime.now():%Y-%m-%d %H:%M}",
        f"- 검문: {verdict}",
        "",
        text[:4000],
    ]
    # 계약서.pdf 와 계약서.html 과 기존 raw/docs/계약서.md 가 같은 이름을 노린다.
    # 남의 문서를 덮어쓰느니 번호를 붙인다 — 사라지는 것보다 두 개가 낫다.
    out = os.path.join(dest, name + ".md")
    n = 2
    while os.path.exists(out):
        out = os.path.join(dest, f"{name}-{n}.md")
        n += 1
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(note))

    return {
        "file": os.path.basename(path),
        "risks": len(risks),
        "verdict": "BLOCK" if risks else "PASS",
        "saved": os.path.relpath(out, BRAIN_HOME),
        "detail": risks,
    }


def main(argv) -> int:
    if not argv:
        print(__doc__.strip().splitlines()[0])
        print("\n사용: python3 extensions/w10_vision_ingest.py <파일> [파일...]")
        print(f"현재 BRAIN_HOME: {BRAIN_HOME}")
        return 1

    print(f"[vision_ingest] BRAIN_HOME={BRAIN_HOME}")
    blocked = 0
    for path in argv:
        if not os.path.isfile(path):
            print(f"  ✗ {path}: 파일이 없습니다")
            continue
        try:
            r = vision_ingest(path)
        except RuntimeError as exc:          # 도구 부재는 이유를 그대로 보여준다
            print(f"  ✗ {os.path.basename(path)}: {exc}")
            continue
        mark = "⛔" if r["verdict"] == "BLOCK" else "✓"
        print(f"  {mark} {r['file']}  [{r['verdict']}]  → {r['saved']}")
        for kind, snippet in r["detail"]:
            print(f"      · {kind}: {snippet[:60]}")
        blocked += r["verdict"] == "BLOCK"

    if blocked:
        print(f"\n{blocked}건이 review/ 로 갔습니다. 내용을 확인하고 직접 판단하세요.")
    print("\n다음: python3 scripts/compile.py  (raw/ 에 들어온 문서를 위키로)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
