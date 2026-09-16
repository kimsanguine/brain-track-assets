"""brain_mcp.py — 내 브레인을 MCP로 노출한다 (이식 ⑤).

핵심 설계: **읽기 도구는 열고, 쓰기 도구는 승인받는다.**
  brain_search  (읽기) — 승인 없이 실행
  brain_ingest  (쓰기) — approved=True 없이는 실행하지 않음

연결(Claude Desktop / Codex 설정 예):
  {"mcpServers": {"my-brain": {"command": "python", "args": ["<이 파일의 절대경로>"]}}}
"""
import os

WIKI = os.environ.get("BRAIN_WIKI", "./seed_wiki")

try:
    from fastmcp import FastMCP
except ImportError:
    FastMCP = None

if FastMCP:
    mcp = FastMCP("my-brain")

    @mcp.tool()
    def brain_search(query: str, top_k: int = 3) -> list:
        """내 위키에서 관련 메모를 찾는다. 읽기 전용이라 승인 없이 쓸 수 있다."""
        ...

    @mcp.tool()
    def brain_ingest(title: str, body: str, approved: bool = False) -> dict:
        """내 위키에 새 메모를 저장한다. **승인 없이는 저장하지 않는다.**"""
        if not approved:
            return {"상태": "승인대기", "이유": "쓰기 도구는 사람 승인이 필요합니다"}
        ...

    if __name__ == "__main__":
        mcp.run()
else:
    print("fastmcp 미설치 — 구조만 확인하고 CP5(로컬 호출 로그)로 가세요.")
