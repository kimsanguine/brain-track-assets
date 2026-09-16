"""ocr_lib — 10주차 공용 OCR 도구상자 (노트북·교안이 함께 쓴다).

Block 계약: {"text": str, "box": [x1,y1,x2,y2], "conf": 0~1}
전처리는 '조합'이며, 조합마다 결과가 다르다 — 그래서 재본다.
"""
from __future__ import annotations
import difflib
import numpy as np
from PIL import Image, ImageOps, ImageFilter

try:
    import pytesseract
    HAS_TESS = True
except Exception:                                   # pragma: no cover
    HAS_TESS = False

# ── 정답지: 사람이 눈으로 읽은 값 (채점 기준) ──────────────────────
TRUTH = {
    "회원명": "김수현",
    "결제일": "2026-09-25",
    "청구금액": "384,200",
    "이용기간": "2026-08-01~2026-08-31",
    "합계": "384,200",
}

# 키 없이도 노트북이 돌게 하는 Mock (tesseract 미설치 대비)
MOCK_BLOCKS = [
    {"text": "행복카드", "box": [210, 238, 486, 311], "conf": 0.93},
    {"text": "이용대금", "box": [500, 235, 760, 310], "conf": 0.91},
    {"text": "명세서", "box": [770, 233, 960, 308], "conf": 0.90},
    {"text": "김수현", "box": [388, 402, 520, 447], "conf": 0.88},
    {"text": "2026-09-25", "box": [385, 690, 640, 738], "conf": 0.86},
    {"text": "384,200", "box": [729, 841, 943, 896], "conf": 0.91},
]


def to_blocks(img: Image.Image, lang: str = "kor+eng", psm: int = 6) -> list[dict]:
    """이미지 → Block 리스트. tesseract가 없으면 Mock을 돌려준다(연결 먼저 확인)."""
    if not HAS_TESS:
        return [dict(b) for b in MOCK_BLOCKS]
    data = pytesseract.image_to_data(img, lang=lang, config=f"--psm {psm}",
                                     output_type=pytesseract.Output.DICT)
    out = []
    for i, t in enumerate(data["text"]):
        t = t.strip()
        conf = float(data["conf"][i])
        if not t or conf < 0:
            continue
        out.append({"text": t,
                    "box": [data["left"][i], data["top"][i],
                            data["left"][i] + data["width"][i],
                            data["top"][i] + data["height"][i]],
                    "conf": round(conf / 100, 2)})
    return out


# ── 전처리 3종 (+대비 보정) ───────────────────────────────────────
def autocontrast(img):        # 조명 불균일 펴기
    return ImageOps.autocontrast(ImageOps.grayscale(img), cutoff=2)


def denoise(img):             # 얼룩(노이즈) 지우기
    return img.filter(ImageFilter.MedianFilter(3))


def upscale(img, k: int = 2):  # 확대 — 작은 글자에 픽셀을 준다
    return img.resize((img.width * k, img.height * k), Image.LANCZOS)


def deskew(img, lo=-8.0, hi=8.0, step=0.5):
    """기울기 자동 보정. 각도 추정은 축소본에서(빠르게), 적용은 원본에서(선명하게).

    원리: 글줄이 수평이면 가로 방향 검은 픽셀 합의 '들쭉날쭉함'이 최대가 된다.
    반환: (보정된 이미지, 찾은 각도)
    """
    g = ImageOps.grayscale(img)
    small = g.resize((max(80, g.width // 3), max(80, g.height // 3)), Image.BILINEAR)
    arr = np.array(small)
    # 여백 제외: 글자가 있는 가운데만 본다
    h, w = arr.shape
    arr = arr[int(h * .08):int(h * .92), int(w * .08):int(w * .92)]
    base = Image.fromarray(arr)
    best, best_score = 0.0, -1.0
    for a in np.arange(lo, hi + 1e-9, step):
        rot = np.array(base.rotate(a, resample=Image.BILINEAR, fillcolor=255))
        ink = (255 - rot).astype(np.float64)
        rows = ink.sum(axis=1)
        score = float(np.var(np.diff(rows)))      # 줄 경계가 뚜렷할수록 큼
        if score > best_score:
            best, best_score = float(a), score
    fill = 255 if img.mode == "L" else (255, 255, 255)
    return img.rotate(best, resample=Image.BICUBIC, fillcolor=fill), round(best, 1)


def otsu(img):                # 이진화 — 흑백으로 딱 자르기
    g = np.array(ImageOps.grayscale(img))
    hist, _ = np.histogram(g, 256, [0, 256])
    total, sum_all, wB, sB, mx, th = g.size, float(np.dot(np.arange(256), hist)), 0, 0.0, 0.0, 127
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sB += t * hist[t]
        var = wB * wF * ((sB / wB) - ((sum_all - sB) / wF)) ** 2
        if var > mx:
            mx, th = var, t
    return Image.fromarray(((g > th) * 255).astype("uint8"))


# ── 채점: 정답지와 얼마나 닮았나 (0~1) ────────────────────────────
def field_scores(blocks: list[dict], truth: dict | None = None) -> dict:
    truth = truth or TRUTH
    joined = "".join(b["text"] for b in blocks).replace(" ", "")
    out = {}
    for k, v in truth.items():
        target = v.replace(" ", "")
        if target in joined:
            out[k] = 1.0
            continue
        # 부분 점수: 정답과 가장 닮은 같은 길이 구간을 찾는다
        best = 0.0
        n = len(target)
        for i in range(0, max(1, len(joined) - n + 1)):
            r = difflib.SequenceMatcher(None, target, joined[i:i + n]).ratio()
            if r > best:
                best = r
        out[k] = round(best, 2)
    return out


def grade(blocks: list[dict], truth: dict | None = None) -> dict:
    fs = field_scores(blocks, truth)
    exact = sum(1 for v in fs.values() if v >= 0.999)
    return {"fields": fs,
            "exact": exact,
            "total": len(fs),
            "avg": round(float(np.mean(list(fs.values()))), 2),
            "blocks": len(blocks),
            "conf": round(float(np.mean([b["conf"] for b in blocks])), 2) if blocks else 0.0}
