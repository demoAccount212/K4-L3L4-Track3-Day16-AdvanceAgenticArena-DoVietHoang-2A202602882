"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông "chính thống" đó —
dù mỗi câu được lấy nguyên văn từ một tài liệu khác.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG khớp NGUYÊN VĂN một DÒNG nào trong
    corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Chú ý chữ DÒNG: kiểm tra `claim["text"] in doc.body` (cả khối, không
tách dòng) là SAI — scorer chỉ nhận trích dẫn khớp nguyên văn MỘT DÒNG
(xem "ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP" ngay dưới đây). `in doc.body` coi
một câu vắt qua hai dòng là hợp lệ, trong khi scorer thì không — tín
hiệu kiểu đó khiến bạn giữ nguyên một trích dẫn mà scorer vẫn chấm
`HALLUCINATED`.

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§11). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bất kỳ bằng chứng nào -> BỊA, để `critic` xoá.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu là trích dẫn
    nguyên văn của MỘT DÒNG trong tài liệu được trích VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi dấu nháy, "chuẩn hoá" khoảng trắng,
    hay vá lại câu bị cắt bằng nội dung lấy từ corpus đều làm mất cả hai
    điều kiện cùng lúc (đo được: -40 điểm).

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa: điều kiện
`doc.body in ctx.observed_text` nghĩa là "tài liệu này đã về nguyên vẹn
từ một lần fetch sạch" — một đoạn snippet hay một bản bị cắt không tính.

CÔNG CỨ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from harness.middleware import Middleware


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def _line_contains_text(self, line: str, text: str) -> bool:
        """Check if text appears verbatim in line (substring match after strip)."""
        return text.strip() in line

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not claims or not isinstance(claims, list):
            return report

        corrected: list[dict] = []
        updated_citations: set[str] = set()

        for claim in claims:
            text = claim.get("text", "")
            doc_id = claim.get("doc_id", "")

            # Bước 1: Nếu tài liệu tồn tại VÀ claim["text"] khớp NGUYÊN VĂN
            # một DÒNG trong body của nó -> trích dẫn đã đúng, giữ nguyên claim.
            doc = ctx.corpus.get(doc_id) if ctx.corpus else None
            if doc and doc.body:
                for line in doc.body.splitlines():
                    if self._line_contains_text(line, text):
                        corrected.append(claim)
                        updated_citations.add(doc_id)
                        break
                else:
                    # Không tìm thấy dòng khớp trong doc body -> bước 2
                    pass
            else:
                # Bước 1 thất bại (doc không tồn tại hoặc body rỗng) -> bước 2
                pass

            # Bước 2: Nếu chưa giữ claim -> tìm trong ctx.corpus.docs
            # tài liệu đầu tiên thoả doc.body in ctx.observed_text
            # VÀ claim["text"] khớp nguyên văn một DÒNG của doc.body
            if not any(c is claim for c in corrected):
                found_doc_id = None
                if ctx.corpus and ctx.corpus.docs:
                    for candidate in ctx.corpus.docs:
                        candidate_body = candidate.body or ""
                        # Kiểm tra: doc.body in ctx.observed_text
                        if candidate_body in ctx.observed_text:
                            # Kiểm tra claim["text"] khớp một DÒNG của doc.body
                            for line in candidate_body.splitlines():
                                if self._line_contains_text(line, text):
                                    found_doc_id = candidate.doc_id
                                    break
                            if found_doc_id:
                                break

                if found_doc_id:
                    # Đổi doc_id sang nó, GIỮ NGUYÊN text
                    corrected.append({"text": text, "doc_id": found_doc_id})
                    updated_citations.add(found_doc_id)
                # Nếu không tìm được -> để claim như cũ, để `critic` xử lý

        # Cập nhật report
        report["claims"] = corrected
        if updated_citations:
            report["citations"] = sorted(updated_citations)
        else:
            report["citations"] = []

        return report