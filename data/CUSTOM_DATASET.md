# Custom dataset — ticket hỗ trợ CNTT nội bộ doanh nghiệp (Bonus B2)

## Nguồn

Sinh tổng hợp (synthetic), **không dùng dữ liệu thật của bất kỳ tổ chức nào** —
`scripts/make_custom_dataset.py`, seed cố định `20260824` để tái lập được. Không có bước
"khử nhiễu" vì không có PII thật để loại bỏ: mọi tên phòng ban, thiết bị, câu chữ đều là
mẫu tổng hợp từ danh sách từ vựng viết tay, không tham chiếu tới người/công ty thật.

## Vì sao đổi miền, không chỉ đổi seed

Corpus mặc định của lab là ticket CSKH thương mại điện tử. Bộ này đổi sang **ticket hỗ trợ
CNTT nội bộ doanh nghiệp** (nhân viên báo lỗi phần mềm/phần cứng, xin cấp quyền, xin cài
đặt, hỏi hướng dẫn sử dụng) — một miền nghiệp vụ khác hẳn, không phải chỉ đổi tên sản phẩm
trong cùng một khuôn câu. Vẫn giữ nguyên schema 4 khoá (`intent`, `urgency`, `product`,
`sentiment`) để tương thích toàn bộ code chấm điểm có sẵn trong `src/labkit/evaluate.py`
mà không cần sửa dòng nào — chỉ có **giá trị** trong 4 trường và câu hỏi (`instruction`)
đổi theo miền mới.

| Trường | Miền mặc định (CSKH) | Miền mới (IT nội bộ) |
|---|---|---|
| `intent` | doi_tra, van_chuyen, hoan_tien, san_pham_loi, hoi_thong_tin | loi_phan_mem, loi_phan_cung, cap_quyen, cai_dat_moi, huong_dan_su_dung |
| `product` | 12 mặt hàng thương mại điện tử | 15 hệ thống/thiết bị CNTT nội bộ (VPN, email công ty, máy chủ, phần mềm CRM...) |
| `urgency`, `sentiment` | 3 mức mỗi loại | giữ nguyên 3 mức, đổi cụm từ diễn đạt |

## Kích thước

| File | Số dòng |
|---|---|
| `data/train_seed.jsonl` | 260 |
| `data/eval_target.jsonl` | 50 |
| `data/holdout_secret.jsonl` | 30 |
| `data/eval_regression.jsonl` | 15 (**giữ nguyên bản gốc**, xem lý do bên dưới) |

≥200 mẫu train theo yêu cầu B2 (260 mẫu).

## Vì sao `eval_regression.jsonl` không đổi

Nhóm regression đo **năng lực chung** (kiến thức phổ thông, không liên quan tác vụ triage)
để bắt catastrophic forgetting — mục đích của nó không phụ thuộc miền của tác vụ chính, nên
15 câu hỏi kiến thức phổ thông gốc vẫn hợp lệ và không cần thay khi đổi miền triage.

## Chất lượng / đa dạng

Khác với generator mặc định (một khuôn câu cố định, chỉ thay từ), generator này:
- Nhiều biến thể cụm từ cho mỗi giá trị field (3-5 cách diễn đạt/trường)
- Thứ tự mệnh đề (urgency/sentiment) được xáo ngẫu nhiên mỗi mẫu, không cố định trước-sau
- 3 kiểu câu mở đầu + 3 connector khác nhau, kết hợp ngẫu nhiên
- Kiểm tra khử trùng lặp (dedup theo nội dung) và khử nhiễm chéo train/eval (assert không
  câu ticket nào xuất hiện cả ở train lẫn eval) ngay trong script

## Tái lập

```bash
python scripts/make_custom_dataset.py
```

Sinh lại đúng 3 file (`train_seed.jsonl`, `eval_target.jsonl`, `holdout_secret.jsonl`)
từ seed cố định, không đụng `eval_regression.jsonl`. Sau khi chạy, phải chạy lại **toàn bộ
pipeline NB1→NB5** (và NB4/NB6/rank_sweep nếu muốn giữ bonus khác) vì mọi mốc đã đóng băng
trước đó (baseline, adapter, verdict) được đo trên corpus cũ, không còn hợp lệ để so sánh.
