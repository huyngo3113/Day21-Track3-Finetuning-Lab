# Lab 21 — Evaluation Report

**Họ tên**: Ngô Mạnh Minh Huy  **MSSV**: 2A202601926  **Ngày**: 2026-08-22
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 (sm_75, 14.6 GB, fp16)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Trạng thái nộp bài**: NB1→NB5 chạy full trên Colab T4 (`EVAL_LIMIT` bỏ trống, n=50
> target / n=15 regression). `results/*.json`, `runs.csv` đã push lên repo. Cả hai bonus
> B1 (NB6 merge + hot-swap) và B4 (quét rank r∈{8,16,64}) đã chạy và push. 3 unit test
> từng FAIL vì bug import trong `tests/test_env_and_silent_defaults.py` (`from
> tests.fake_tokenizer import ...` thay vì `from fake_tokenizer import ...` như mọi file
> test khác trong repo dùng) — đã sửa, xác nhận pass local. `adapters/correct/*.safetensors`
> (123.91 MB) vượt giới hạn 100MB của GitHub — không push kèm repo; adapter config/tokenizer
> vẫn có trong `adapters/correct/`, trọng số đầy đủ đang chờ push lên HuggingFace Hub (B5).

---

## 1. Setup

| | |
|---|---|
| Dataset | seed corpus mặc định — 250 ticket CSKH → JSON triage (4 khoá: intent, urgency, product, sentiment) |
| Train / val | 225 / 25 (seed 42) — *(data/split)* |
| `max_length` | tier T4 mặc định = 1024; p95 đo được = **98** token (max=101) → `suggested_max_length=256` *(results/token_stats.json)*. Ghi chú: 1024 rộng hơn nhiều so với p95 vì đây là default tier chưa chỉnh; không cắt mất mẫu nào (max quan sát 101 < 256) nhưng tốn VRAM/thời gian hơn cần thiết — có thể hạ xuống 256 ở lần chạy sau. |
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

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

Full run, `EVAL_LIMIT` bỏ trống — n=50 target, n=15 regression — *(results/baselines_frozen.json, results/verdict.json)*.

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7578 | 0.000 | 3146.4 |
| (b) base + optimized prompt | 0.765 | 0.7578 | 1.000 | 976.9 |
| (c) LoRA fine-tune | 0.970 | 0.6333 | 1.000 | 1425.8 |

**(b) có thật sự mạnh hơn (a) không?** **Có** — 0.765 vs 0.000 trên target, và format nhảy
từ 0 lên 1.0 (prompt tối ưu ép model trả JSON đúng khuôn, ngay cả khi nội dung còn sai).
`verify.py` xác nhận `baseline (b) beats (a)` PASS và `baseline (b) prompt unmodified`
PASS — `OPTIMIZED_PROMPT` dùng nguyên bản gốc repo, không chỉnh sửa (SHA `719e74d3b6232053`).

---

## 4. Giải phẫu cấu hình sai (NB4)

Full run — *(results/autopsy.json, results/runs.csv)*.

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6263 | 0.970 | 924.5 | 12.01 |
| `attn_only` | q,v (matched) | 283 | 32,456,704 | 1e-4 | 0.5385 | 0.970 | 795.9 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.000 | 932.6 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 1001.0 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

`attn_only` được `verify.py` xác nhận là đối chứng công bằng: 32,456,704 tham số huấn
luyện so với 32,464,896 của `correct`, lệch < 0.03% (< 5% ngưỡng yêu cầu). Trên tập
target đầy đủ (n=50) chúng **hoà tuyệt đối** — cả hai đạt 0.970. Nhưng train loss lại
KHÔNG đồng thuận: `attn_only` kết thúc ở 0.5385, thấp hơn hẳn 0.6263 của `correct` — nếu
xếp hạng bằng train loss (đúng lỗi 2.5 mà rubric cảnh báo) sẽ kết luận `attn_only` "tốt
hơn", trong khi trên chỉ số thật (target accuracy) chúng ngang nhau tuyệt đối. `attn_only`
còn nhanh hơn (795.9s vs 924.5s train, 901.5ms vs 1425.8ms latency) vì có ít module hơn để
forward/backward qua. Kết luận: ở ngân sách 32M tham số và 30 step này, **rank** (khớp
ngân sách tham số) là đòn bẩy quyết định độ chính xác, không phải **vị trí gắn adapter**
(chỉ q,v so với toàn bộ 12 lớp linear) — khi đã match ngân sách, vị trí gắn không tạo
khác biệt đo được trên tác vụ target, dù có tạo khác biệt trên train loss và tốc độ.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

LR bị hạ 10 lần (1e-5 thay vì 1e-4, đúng thang full-fine-tune thay vì thang LoRA). Loss
giảm rất chậm và không hội tụ: từ 2.163 xuống chỉ còn 1.119 sau 30 step (so với `correct`
rơi từ 2.163 xuống 0.024) — `mean_token_accuracy` dừng ở ~0.79, không bao giờ vượt 0.99
như các run khác. Nếu chỉ nhìn con số loss cuối (1.5704) mà không biết LR, sẽ dễ kết luận
nhầm "model học chậm nhưng vẫn học được, chỉ cần train thêm step" — thực tế đo trên target
đầy đủ (n=50) lại là **0.000, sập hoàn toàn**, format cũng 0.000, và latency tăng vọt lên
**5087.7ms** (gấp 5.5 lần `correct`) vì model sinh ra output dài, lộn xộn, không đúng
format JSON, phải chạy hết max token mới dừng. Loss "trông có vẻ ổn" che giấu một model đã
hỏng hoàn toàn ở tác vụ thật — đây chính là lý do rubric cấm dùng train loss để xếp hạng,
và latency bất thường (5x) đôi khi là tín hiệu sớm hơn cả target score.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

`qlora` dùng 7.09 GB VRAM đỉnh so với 12.01/12.02 GB của `correct`/`attn_only` — tiết kiệm
4.92-4.93 GB (~41%). Đổi lại, target rơi từ 0.970 xuống 0.940 (mất 0.030, ~3% tương đối),
format vẫn giữ 1.000. Thời gian train cũng nhích lên (1001.0s vs 924.5s, +8%) do overhead
quantize/dequantize, và latency inference cao hơn (1747.1ms vs 1425.8ms). Số đo đầy đủ
(n=50) cho thấy cái giá về accuracy **nhỏ hơn** ước tính ban đầu từ mẫu nhỏ — chỉ 0.030,
không phải mức lớn. Điều này **ủng hộ một phần, không tuyệt đối** khuyến nghị thận trọng
với QLoRA trên Qwen3.5: có đánh đổi thật (VRAM giảm, tốc độ train chậm hơn, target giảm
nhẹ) nhưng đánh đổi accuracy khá nhỏ so với mức tiết kiệm VRAM lớn. Với tier T4 (16GB) vốn
đã đủ VRAM cho `correct` (12.01GB < 16GB), không có lý do bắt buộc phải đánh đổi; nhưng
nếu VRAM là nút thắt thật sự (tier LAPTOP 8GB), 3% target đổi lấy 41% VRAM là một đánh đổi
hợp lý, không phải "cấm tuyệt đối" như khuyến nghị vendor gợi ý.

**4.4 (Bonus B4) — Quét rank có kiểm soát, cố định vị trí = text-linear, r ∈ {8, 16, 64}.
Khi nào rank mới là đòn bẩy?** *(results/rank_sweep.json)*

| r | trainable | target | format | latency (ms) |
|---|---|---|---|---|
| 8 | 16,232,448 | 0.87 | 1.0 | 1403.9 |
| 16 (`correct`) | 32,464,896 | 0.97 | 1.0 | 1425.8 |
| 64 | 129,859,584 | 1.00 | 1.0 | 1325.9 |

Ở **cùng một vị trí** (text-linear, không đổi placement như 4.1), target tăng đơn điệu
theo rank: 0.87 → 0.97 → 1.00. Điều này trả lời trực tiếp câu hỏi B4: **rank là đòn bẩy
thật khi vị trí gắn adapter đã cố định** — tăng rank tăng năng lực khớp dữ liệu train.
Nhưng lợi ích giảm dần rõ rệt: r=8→16 (gấp đôi tham số) được +0.10 target; r=16→64 (gấp
4 lần tham số) chỉ được thêm +0.03 và đã chạm trần 1.00 trên 50 mẫu target — nhiều khả
năng là dấu hiệu ghi nhớ (memorization) trên tập train nhỏ (225 mẫu) hơn là năng lực tổng
quát hoá thật, vì regression không được đo lại ở các rank này (script B4 chỉ chấm target,
xem "Nếu có thêm 2 giờ nữa"). Kết hợp với 4.1 (vị trí không đổi kết quả khi đã khớp ngân
sách rank): bức tranh đầy đủ là — **rank quyết định năng lực, vị trí gắn không quyết định
(ở ngân sách đủ lớn)**, nhưng rank cao hơn không miễn phí — cần đo regression trước khi kết
luận r=64 "tốt hơn" r=16 cho triển khai thật.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = -0.124` · `valid_trace_rate = 0.00`

Lý do gate ghi lại: *"general capability regressed by 0.124 (tolerance 0.020). See deck
§14.3 — add 1-5% replay data."*

Fine-tune (c) thắng rõ trên target: 0.970 vs 0.765 của baseline (b) đã prompt tối ưu
(Δ=+0.205, lớn hơn nhiều so với ngưỡng cần để coi là cải thiện thật). Nhưng đồng thời làm
**regression sập từ 0.7578 xuống 0.6333** (Δ=-0.1244), vượt xa dung sai ±0.020 mà cổng cho
phép — đây là **catastrophic forgetting** kinh điển: 250 mẫu train chỉ tập trung vào một
tác vụ hẹp (triage JSON) khiến model quên một phần năng lực kiến thức/chỉ dẫn phổ thông đo
bằng 15 câu hỏi regression. `valid_trace_rate=0.00` không phải chỉ số gây FAIL ở đây —
corpus mặc định không có suy luận thật trong `<think>` nên chỉ số này không áp dụng, gate
FAIL hoàn toàn do regression, không liên quan reasoning trace.

Kết quả FAILED này **đáng tin và có thể giải thích được**, không phải lỗi pipeline: mask
proof xanh, baseline (b) đã prompt tối ưu và không bị làm yếu, bốn run cùng step budget,
`attn_only` là đối chứng công bằng — mọi điều kiện thí nghiệm đều đúng. Nguyên nhân gốc là
**thiết kế dữ liệu train**: 225 mẫu train 100% là ticket CSKH, không có "replay data" (dữ
liệu kiến thức phổ thông trộn vào 1-5% như deck §14.3 khuyến nghị) để giữ năng lực gốc.
Nếu chỉ nhìn target Δ dương mà bỏ qua regression, sẽ kết luận nhầm "fine-tune thành công" —
đây chính xác là bẫy mà thiết kế bốn-nhóm của lab được dựng ra để bắt.

---

## 6. Định tính — bắt buộc có cả ca THUA

Trích từ `results/qualitative.json` (full n=50). Trường `ft_pred` bị cắt ở ~90 ký tự trong
log gốc (file đầy đủ 50/50 mục, xác nhận qua `results/qualitative.json`). Điểm số
(`ft_score`) là tỷ lệ 4 trường đúng/4 — toàn bộ 50 mục chỉ có hai giá trị: 1.0 (44 mục) và
0.75 (6 mục, đúng 3/4 trường), không có mục nào 0.0/0.25/0.5 — mọi lỗi của fine-tune đều
là sai đúng một trường, chưa từng sai hoàn toàn cả 4. Không có cột "(b) prompt" và "nhãn
đúng" đầy đủ trong export này — `qualitative.json` chỉ lưu `i, ticket, ft_score, ft_pred`;
xem `data/eval_target.jsonl` dòng tương ứng `i` để đối chiếu nhãn gốc khi cần chi tiết hơn.

| # | Ticket (rút gọn) | ft_score | Dự đoán fine-tune (rút gọn) | Nhận xét |
|---|---|---|---|---|
| i=3 | "...bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền." | 0.75 | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": ...}` | ❌ **FT thua** (1/4 trường sai) |
| i=5 | "...nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện." | 0.75 | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sen...}` | ❌ **FT thua** (1/4 trường sai) |
| i=12 | "...áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện." | 0.75 | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment"...}` | ❌ **FT thua** (1/4 trường sai) |
| i=39 | "...nồi chiên không dầu mã đơn VN949966. Hoàn tiền." | 0.75 | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentim...}` | ❌ **FT thua** (1/4 trường sai) |
| i=0 | "...chuột không dây mã đơn VN232232. Cho tôi trả lại." | 1.0 | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_c...}` | ✅ FT thắng |
| i=4 | "...đèn bàn LED mã đơn VN339109. Vỡ khi nhận. Gấp." | 1.0 | `{"intent": "san_pham_loi", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "trung...}` | ✅ FT thắng |
| i=49 | "...ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm nhé." | 1.0 | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sent...}` | ✅ FT thắng |

Có mẫu chung nào ở các ca FT thua không? Toàn bộ tập target (n=50) có đúng **6 ca thua**
(i=3, 5, 12, 39, 41, 46 — 4 ca đầu ở bảng trên, cộng i=41 "giao hàng chậm" và i=46 "sai
màu"), tất cả đều điểm 0.75 (sai đúng 1/4 trường, không có ca nào sai từ 2 trường trở lên).
Điểm chung: đều là ticket **có nhiều khả năng dẫn tới nhầm `intent`** giữa các nhãn gần
nghĩa — "chưa thấy tiền" (`hoan_tien` hay `van_chuyen`?), "thiếu phụ kiện"/"bị lỗi"
(`san_pham_loi` hay `doi_tra`?), "giao hàng chậm" (`van_chuyen` rõ nhưng có thể lẫn
`hoi_thong_tin`) — tức các trường hợp ranh giới nhãn mờ, cần suy luận ngữ cảnh thay vì khớp
từ khóa trực tiếp ("trả lại" → `doi_tra` rõ ràng). Giả thuyết: 250 mẫu train chưa đủ đa
dạng các ca biên giữa `hoan_tien`/`van_chuyen`/`san_pham_loi`/`doi_tra` khi tín hiệu ngữ
nghĩa yếu. Đáng chú ý: **không ca nào sai hoàn toàn** (0.0) — model luôn bắt đúng ít nhất
3/4 trường, nhất quán với target trung bình rất cao (0.970).

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).**

Với dữ liệu và cấu hình đo được, **không nên deploy bản fine-tune `correct` này ở dạng
hiện tại** — cổng hồi quy bốn-nhóm FAILED, không phải vì lỗi pipeline mà vì catastrophic
forgetting thật: target tăng mạnh (+0.205, từ 0.765 lên 0.970) nhưng regression sập
-0.124 (từ 0.7578 xuống 0.6333), vượt xa dung sai ±0.020. Một hệ thống production không
thể chấp nhận đánh đổi năng lực chung để lấy accuracy trên một tác vụ hẹp — model sẽ trả
lời sai các câu hỏi ngoài phạm vi triage mà trước đó nó trả lời đúng. Hướng sửa rõ ràng
theo deck §14.3: trộn 1-5% dữ liệu kiến thức phổ thông vào tập train rồi chạy lại toàn bộ
pipeline để xác nhận regression được giữ trong dung sai mà không hy sinh nhiều target.
Đòn bẩy thật sự trong lab này, tách bạch theo từng thí nghiệm: **rank/ngân sách tham số**
quyết định độ chính xác trên tác vụ (không phải vị trí gắn adapter — `attn_only` hoà tuyệt
đối với `correct` khi ngân sách khớp); **learning rate đúng thang** là điều kiện sống còn
tuyệt đối (`wrong_lr` sập về 0 chỉ vì đổi một con số); và **thành phần dữ liệu train**
(thiếu replay) là nguyên nhân trực tiếp của thất bại ở cổng hồi quy — ba đòn bẩy này độc
lập với nhau, và bài học lớn nhất của lab là target tăng không đồng nghĩa "nên deploy" nếu
không kiểm tra đủ bốn nhóm.

**Ba điều tôi học được** (cụ thể, không generic):
1. Train loss thấp không đồng nghĩa target accuracy cao — `attn_only` có train loss thấp
   hơn `correct` (0.5385 vs 0.6263) nhưng target accuracy bằng nhau tuyệt đối (0.970);
   chấm điểm bằng loss thay vì chỉ số tác vụ thật (rubric 2.5) sẽ ra kết luận sai.
2. Một fine-tune "thắng" rõ trên metric chính (target +0.205) vẫn có thể là một thất bại
   toàn cục nếu không đo song song năng lực chung — nếu lab này chỉ có một baseline target
   mà không có nhóm regression, tôi đã kết luận nhầm "PASS" và đề xuất deploy.
3. Một sai số LR (10 lần) không "làm chậm học" mà làm **sập hoàn toàn** cả target lẫn
   format, và còn để lộ dấu hiệu qua latency tăng vọt (5087.7ms, gấp 5.5 lần) — dấu hiệu dễ
   quan sát hơn cả việc chờ tính target score.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 1-5% dữ liệu phổ thông (câu hỏi kiến thức
chung, không liên quan CSKH) vào 225 mẫu train, chạy lại `correct` với đúng cấu hình hiện
tại, và so sánh trực tiếp regression Δ trước/sau replay — mục tiêu đưa Δ về trong dung sai
±0.020 mà vẫn giữ phần lớn mức tăng target +0.205. Sau đó thử B4 (quét rank có kiểm soát
r ∈ {8,16,64} ở text-linear) để trả lời câu hỏi rank thật sự cần bao nhiêu là đủ cho tác vụ
này, vì `attn_only` đã cần nâng lên r=283 chỉ để hoà `correct` ở r=16.

---

## Phụ lục — thưởng đã làm

### B1 — NB6: merge + assert không tụt điểm + hot-swap

*(results/merge_check.json)*

| | trước merge | sau merge | Δ |
|---|---|---|---|
| target (n=50) | 0.9700 | 0.9700 | +0.0000 |

Merge `W = W₀ + (α/r)·BA` giữ nguyên tuyệt đối điểm target — không tụt (ngưỡng cho phép
±0.01, đạt Δ=0.0000). Đã hot-swap thành công `correct` trên cùng một base model đang nạp
trong VRAM (phần 3 của NB6, sau khi sửa lỗi rò VRAM giữa bước merge và bước hot-swap —
xem `notebooks/06_merge_and_serve.py`, commit sửa lỗi `del model` thiếu trước khi load
base mới).

### B4 — Quét rank có kiểm soát

Xem mục 4.4 ở trên — r ∈ {8, 16, 64} tại cùng vị trí text-linear, target tăng đơn điệu
0.87 → 0.97 → 1.00, lợi ích giảm dần theo rank.

- [x] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [x] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
