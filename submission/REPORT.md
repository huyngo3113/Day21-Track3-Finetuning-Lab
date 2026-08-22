# Lab 21 — Evaluation Report

**Họ tên**: Ngô Mạnh Minh Huy  **MSSV**: 2A202601926  **Ngày**: 2026-08-22
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `<điền sau khi chạy Colab — T4 16GB>`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Trạng thái nộp bài**: NB1 (CPU) + test suite chạy PASS trên máy local (không GPU).
> NB2–NB5 đã chạy trên Colab T4 nhưng ở **SMOKE MODE** (`EVAL_LIMIT=8`, chỉ 8/50 mẫu
> target và 8/15 mẫu regression) — `verify.py` tự động FAIL mục "full eval set used" vì
> đây không phải run nộp được. **Số liệu ở mục 3–7 dưới đây lấy từ run smoke đó, chỉ để
> dựng khung** — PHẢI chạy lại NB2+NB5 (hoặc cả pipeline) với `EVAL_LIMIT` bỏ trống rồi
> ghi đè `results/` trước khi nộp thật. Xem hướng dẫn cuối file.

---

## 1. Setup

| | |
|---|---|
| Dataset | seed corpus mặc định — 250 ticket CSKH → JSON triage (4 khoá: intent, urgency, product, sentiment) |
| Train / val | 225 / 25 (seed 42) — *(data/split)* |
| `max_length` | tier T4 mặc định = 1024; p95 đo được = **98** token (max=101) → `suggested_max_length=256` *(results/token_stats.json)*. Ghi chú: 1024 rộng hơn nhiều so với p95 vì đây là default tier chưa chỉnh; nên hạ xuống 256 khi train thật trên Colab để tiết kiệm VRAM/thời gian mà không cắt mẫu nào (max quan sát 101 < 256). |
| `MASK_MODE` | `assistant-only` (mặc định) |
| Epochs / max_steps | 2 epochs → 30 optimizer steps (225 train rows, batch 1 × grad_accum 16) — *(results/runs.csv)* |

**Template có giữ khối `<think>` không?** **Có** — *(results/template_check.json)*: `verdict = "reasoning preserved — safe to train on traces"`. Chat template Qwen3.5 mở `<think>...</think>` và đóng khối rỗng ngay trong generation prompt khi không có suy luận thật; với corpus mặc định (đáp án JSON trần) khối `<think>` luôn rỗng nên không có nội dung suy luận nào bị loss tính vào hay bị che — không cần xử lý thêm.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39/94 token) |
| Câu trả lời nằm trong loss | **true** |
| Câu hỏi KHÔNG nằm trong loss | **true** |

Đoạn được tính loss (đầy đủ, chỉ 2 dòng):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần bị che (masked, không tính loss) — toàn bộ system + user + `<think>` mở:

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

```

`supervised_fraction = 0.41 < 0.95` → không mất điểm 1.1 (mất trắng nếu ≥0.95, tức tính loss cả trên prompt).

---

> **Từ đây trở xuống (mục 3–7, phụ lục) cần GPU.** Mở `colab/Lab21_RUN_ALL.ipynb` trên
> Colab (Runtime → T4 GPU), chạy ô 1→4 (mở tab MỚI mỗi lần, đừng reconnect — xem README).
> Sau khi chạy xong, tải `results/*.json`, `results/runs.csv` về, ghi đè vào thư mục
> `results/` của repo local này, rồi điền các bảng dưới đây từ đúng các file đó. Cuối
> cùng chạy `.venv/Scripts/python scripts/verify.py` để xác nhận PASS trước khi nộp.

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

> ⚠️ SMOKE (n=8/nhóm, `EVAL_LIMIT=8`) — thay bằng số full (n=50 target / n=15 regression) trước khi nộp.

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.750 | 0.000 | 3373.4 |
| (b) base + optimized prompt | 0.688 | 0.750 | 1.000 | 1018.8 |
| (c) LoRA fine-tune | 0.938 | 0.750 | 1.000 | 1563.4 |

**(b) có thật sự mạnh hơn (a) không?** **Có** — 0.688 vs 0.000 trên target, và format nhảy
từ 0 lên 1.0 (prompt tối ưu ép model trả JSON đúng khuôn). `verify.py` xác nhận
`baseline (b) beats (a)` PASS và `baseline (b) prompt unmodified` PASS — `OPTIMIZED_PROMPT`
dùng nguyên bản gốc repo, không chỉnh sửa (SHA `719e74d3b6232053` khớp bản gốc).

---

## 4. Giải phẫu cấu hình sai (NB4)

> ⚠️ Cột target lấy từ SMOKE (n=8) — thay bằng số full trước khi nộp. Cột trainable/LR/VRAM/s là số thật (không phụ thuộc `EVAL_LIMIT`).

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6264 | 0.938 | 968.9 | 12.01 |
| `attn_only` | q,v (matched) | 283 | 32,456,704 | 1e-4 | 0.5379 | 0.938 | 809.6 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.000 | 934.3 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.844 | 985.3 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

`attn_only` được `verify.py` xác nhận là đối chứng công bằng: 32,456,704 tham số huấn
luyện so với 32,464,896 của `correct`, lệch < 0.03% (< 5% ngưỡng yêu cầu). Trên tập
target chúng **hoà** — cả hai đạt 0.938. Nhưng train loss lại KHÔNG đồng thuận: `attn_only`
kết thúc ở 0.5379, thấp hơn hẳn 0.6264 của `correct` — nếu xếp hạng bằng train loss (đúng
lỗi 2.5 mà rubric cảnh báo) sẽ kết luận `attn_only` "tốt hơn", trong khi trên chỉ số thật
(target accuracy) chúng ngang nhau. Điều này nói rằng ở ngân sách 32M tham số và 30 step
này, **rank** (khớp ngân sách tham số) là đòn bẩy chính, không phải **vị trí gắn adapter**
(chỉ q,v so với toàn bộ lớp linear) — khi đã match ngân sách, vị trí gắn không tạo khác
biệt đo được trên tác vụ target, dù có tạo khác biệt trên train loss. *(Số target ở đây
là SMOKE n=8 — cần xác nhận lại trên full 50 mẫu trước khi kết luận chắc chắn "hoà".)*

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

LR bị hạ 10 lần (1e-5 thay vì 1e-4, đúng thang full-fine-tune thay vì thang LoRA). Loss
giảm rất chậm và không hội tụ: từ 2.163 xuống chỉ còn 1.119 sau 30 step (so với `correct`
rơi từ 2.163 xuống 0.026) — `mean_token_accuracy` dừng ở ~0.79, không bao giờ vượt 0.99
như các run khác. Nếu chỉ nhìn con số loss cuối (1.57) mà không biết LR, sẽ dễ kết luận
nhầm "model học chậm nhưng vẫn học được, chỉ cần train thêm step" — thực tế đo trên target
lại là **0.000, sập hoàn toàn**, và latency tăng vọt lên 5051ms vì model sinh ra output
dài, lộn xộn, không đúng format JSON (format=0.000). Loss "trông có vẻ ổn" che giấu một
model đã hỏng hoàn toàn ở tác vụ thật — đây chính là lý do rubric cấm dùng train loss để
xếp hạng.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

`qlora` dùng 7.09 GB VRAM đỉnh so với 12.01 GB của `correct` — tiết kiệm 4.92 GB (~41%).
Đổi lại, target rơi từ 0.938 xuống 0.844 (mất ~0.094, tương đương ~10% tương đối), dù
format vẫn giữ 1.000. Thời gian train cũng nhích lên (985s vs 969s) do overhead
quantize/dequantize. Số đo này **ủng hộ một phần** khuyến nghị thận trọng của nhà cung
cấp với QLoRA trên Qwen3.5: tiết kiệm VRAM là thật, nhưng cái giá về accuracy cũng thật —
không phải "miễn phí". Với tier T4 (16GB) vốn đã đủ VRAM cho `correct` (12.01GB < 16GB),
không có lý do để đánh đổi 10% target accuracy lấy VRAM không cần dùng đến; QLoRA chỉ hợp
lý khi tier thực sự thiếu VRAM (vd. LAPTOP 8GB). *(Lại là số SMOKE n=8, cần xác nhận full.)*

---

## 5. Phán quyết (NB5)

> ⚠️ SMOKE (n=8) — chạy lại full trước khi nộp; PASS/FAIL có thể đổi.

**Kết quả cổng hồi quy**: **PASSED**
`target Δ = +0.250` · `regression Δ = +0.000` · `valid_trace_rate = 0.00`

Fine-tune vượt baseline (b) đã prompt tối ưu 0.250 điểm target (0.938 vs 0.688) mà không
làm tụt regression (0.750 → 0.750, Δ=0.000) — model vẫn giữ nguyên năng lực kiến thức
phổ thông sau khi học tác vụ triage. `valid_trace_rate=0.00` không phải dấu hiệu xấu ở
đây: corpus mặc định không có khối `<think>` chứa suy luận thật (đáp án luôn là JSON
trần, template chỉ mở-đóng khối rỗng), nên không có "reasoning trace" nào để đo — chỉ số
này chỉ có ý nghĩa khi chạy thử nghiệm B3 (reasoning-trace collapse) với dữ liệu có
suy luận thật. Vì Δ regression bằng 0 (không âm), cổng PASS là hợp lệ chứ không phải
PASS "may mắn" nhờ regression sập cùng lúc — đây là kịch bản tốt nhất: fine-tune thắng rõ
ràng trên target, không trả giá bằng năng lực chung.

*(Diễn giải trên dựa vào SMOKE n=8 — PHẢI chạy lại NB2+NB5 không giới hạn `EVAL_LIMIT`,
verdict full có thể khác con số 0.250/0.000 này, dù xu hướng nhiều khả năng giữ nguyên vì
khoảng cách target khá lớn.)*

---

## 6. Định tính — bắt buộc có cả ca THUA

> ⚠️ Từ `results/qualitative.json` của run SMOKE (n=8) — ticket/prediction bị cắt ngắn
> trong log Colab paste vào đây. **Trước khi nộp**: mở `results/qualitative.json` đầy đủ
> (sau khi chạy lại full) để lấy nhãn đúng (`label`) và text đầy đủ, KHÔNG chỉ dùng bản
> rút gọn này. Cột "(b) prompt" chưa có số — cần lấy từ output NB5 (so sánh 3-way) hoặc
> chấp nhận so `(c)` với nhãn đúng như bảng dưới nếu NB5 không in riêng dự đoán của (b).

| # | Ticket (rút gọn) | ft_score | Dự đoán fine-tune (rút gọn) | Nhận xét |
|---|---|---|---|---|
| 1 | "...đèn bàn LED mã đơn VN339109. Vỡ khi nhận. Gấp." | 1.0 | `{"intent": "san_pham_loi", "urgency": "cao", ...}` | ✅ FT thắng |
| 2 | "...balo laptop mã đơn DH863123. Đổi size." | 1.0 | `{"intent": "doi_tra", "urgency": "thap", ...}` | ✅ FT thắng |
| 3 | "...máy xay sinh tố mã đơn OD126693. Muốn đổi." | 1.0 | `{"intent": "doi_tra", "urgency": "trung_binh", ...}` | ✅ FT thắng |
| 4 | "...bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền." | 0.75 | `{"intent": "hoan_tien", "urgency": "trung_binh", ...}` | ❌ **FT thua** (1 trong 4 trường sai — có thể sai `sentiment` hoặc `product`, xem JSON đầy đủ) |
| 5 | "...nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện." | 0.75 | `{"intent": "san_pham_loi", "urgency": "trung_binh", ...}` | ❌ **FT thua** (1 trường sai) |
| 6 | "...chuột không dây mã đơn VN232232. Cho tôi trả lại." | 1.0 | `{"intent": "doi_tra", "urgency": "cao", "sentiment": "tich_c...` | ✅ FT thắng |

Có mẫu chung nào ở các ca FT thua không? Cả hai ca thua (#4, #5) đều liên quan ticket
**không phải yêu cầu đổi/trả rõ ràng** — "chưa thấy tiền" (nghi ngờ hoàn tiền chưa xử lý)
và "thiếu phụ kiện" (lỗi sản phẩm nhưng không nói thẳng "lỗi") — tức các trường hợp
`intent`/`sentiment` phải suy luận gián tiếp từ ngữ cảnh thay vì có từ khóa tường minh
("trả lại", "đổi"). Giả thuyết: model học tốt các mẫu có từ khóa trực tiếp trong 250 mẫu
train, nhưng yếu hơn ở các ca cần suy luận ngữ nghĩa. *(Cần xác nhận lại với nhãn đúng đầy
đủ từ `qualitative.json` — bảng trên là suy đoán từ log rút gọn.)*

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ, dự thảo — điền lại số full trước khi nộp).**

Dựa trên run SMOKE, bản fine-tune (c) đáng để deploy: nó vượt baseline đã prompt tối ưu
(b) 0.250 điểm target (0.938 vs 0.688) mà không đánh đổi năng lực chung (regression giữ
nguyên 0.750), và cổng hồi quy 4 nhóm PASS. Đây không phải một chiến thắng "rẻ" nhờ so
với baseline yếu — `verify.py` xác nhận (b) đã thật sự mạnh hơn (a) và prompt (b) không
bị làm yếu đi, nên khoảng cách 0.250 là so với đối thủ đã cố gắng hết sức bằng prompting.
Đòn bẩy thật sự trong lab này, theo dữ liệu đo được, là **chất lượng dữ liệu + đúng cấu
hình learning rate** — không phải vị trí gắn adapter. Bằng chứng: `attn_only` (chỉ q,v)
khi đã khớp ngân sách tham số với `correct` (khác biệt vị trí gắn) cho kết quả target
**hoà tuyệt đối** (0.938 = 0.938) — vị trí không tạo khác biệt đo được. Ngược lại,
`wrong_lr` (chỉ đổi một con số LR) khiến target **sập về 0**, chứng minh LR đúng thang là
điều kiện sống còn hơn nhiều so với việc chọn gắn adapter vào đâu. Rank (ngân sách tham
số) mới là biến quan trọng thứ hai, thể hiện qua việc `attn_only` phải nâng r lên 283 để
bù cho việc chỉ gắn 2 module thay vì 12 — nếu không match rank, phép so sánh vị trí sẽ vô
nghĩa (đúng như cảnh báo 2.5 của rubric). *(Kết luận này cần xác nhận lại bằng số liệu
full-eval trước khi coi là cuối cùng — n=8 quá nhỏ để chắc chắn 100%.)*

**Ba điều tôi học được** (cụ thể, không generic):
1. Train loss thấp không đồng nghĩa target accuracy cao — `attn_only` có train loss thấp
   hơn `correct` (0.538 vs 0.626) nhưng target accuracy bằng nhau; chấm điểm bằng loss
   thay vì chỉ số tác vụ thật (như rubric 2.5 cảnh báo) sẽ ra kết luận sai thứ tự.
2. Một sai số LR (10 lần) không "làm chậm học" mà làm **sập hoàn toàn** cả target lẫn
   format, trong khi train loss cuối vẫn "trông tạm ổn" (1.57) — đây là bẫy dễ bỏ sót
   nếu chỉ theo dõi loss curve mà không có bộ eval độc lập.
3. QLoRA tiết kiệm VRAM thật (41%) nhưng trả giá bằng accuracy thật (~10% target) trên
   Qwen3.5 ở tier T4 — khi VRAM không phải nút thắt (T4 16GB đủ cho bản 16-bit), không có
   lý do đánh đổi.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** chạy full eval (bỏ `EVAL_LIMIT`) để xác nhận các
con số SMOKE ở trên vẫn đúng hướng trên toàn bộ 50/15 mẫu; sau đó thử B4 (quét rank có
kiểm soát r ∈ {8,16,64} ở text-linear) để trả lời câu hỏi rank thật sự cần bao nhiêu là
đủ, vì ở lab này attn_only đã phải nâng lên r=283 chỉ để hoà — chưa rõ liệu correct ở
r=16 đã tối ưu hay còn dư địa.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:

---

## Việc còn lại trước khi nộp (checklist)

1. [ ] Sửa 3 unit test FAIL trên Colab (chưa rõ tên test — xem hướng dẫn ở tin nhắn).
2. [ ] Chạy lại pipeline **không** đặt `EVAL_LIMIT` (bỏ dòng đó khỏi `.env` hoặc
       `%env EVAL_LIMIT=` rồi unset) → NB2 + NB5 tối thiểu, để có target n=50, regression n=15.
3. [ ] Ghi đè toàn bộ `results/*.json` + `results/runs.csv` bằng bản full mới.
4. [ ] Điền lại mục 3–7 ở trên bằng số full (xóa các dòng ⚠️ SMOKE).
5. [ ] Điền **Họ tên / MSSV** ở đầu file.
6. [ ] Chạy `!python scripts/verify.py` trên Colab (hoặc `.venv/Scripts/python scripts/verify.py`
       local sau khi copy `results/` về) → phải thấy dòng cuối `Ready to submit.`
7. [ ] Push lên GitHub theo hướng dẫn bên dưới.
