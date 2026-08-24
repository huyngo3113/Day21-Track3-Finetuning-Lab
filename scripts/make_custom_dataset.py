#!/usr/bin/env python3
"""B2 bonus — a custom-domain corpus: internal IT helpdesk ticket triage (Vietnamese).

Same 4-key JSON schema as the shipped default (intent/urgency/product/sentiment), so
every scorer in labkit/evaluate.py works unchanged -- only the DOMAIN and VOCABULARY
move, from e-commerce customer support to internal corporate IT support. This keeps
the swap honest: a genuinely different distribution of tickets, not a relabeled copy
of the same sentences.

Richer template grammar than scripts/make_seed_data.py on purpose (multiple phrasings
per field, randomized clause order, optional connectors) so 250+ generated tickets
don't read as one sentence pattern repeated with find-and-replace.

Regression set is left untouched (data/eval_regression.jsonl, unrelated to the task
domain by design -- it exists to catch catastrophic forgetting of GENERAL capability,
which is domain-agnostic).

Run:  python scripts/make_custom_dataset.py
"""
from __future__ import annotations

import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SEED = 20260824

INTENTS = {
    "loi_phan_mem": [
        "phần mềm bị treo liên tục", "báo lỗi khi mở ứng dụng", "không đăng nhập được",
        "ứng dụng tự thoát ra ngoài", "màn hình trắng khi truy cập",
    ],
    "loi_phan_cung": [
        "máy không lên nguồn", "màn hình bị nhấp nháy", "bàn phím liệt vài phím",
        "máy in kẹt giấy liên tục", "quạt tản nhiệt kêu to bất thường",
    ],
    "cap_quyen": [
        "xin cấp quyền truy cập thư mục chung", "xin cấp lại mật khẩu", "xin thêm vào nhóm quyền",
        "tài khoản bị khoá cần mở lại", "xin quyền truy cập từ xa",
    ],
    "cai_dat_moi": [
        "xin cài phần mềm mới cho máy", "cần cấp máy mới cho nhân viên", "xin nối thêm màn hình phụ",
        "cần thiết lập email cho nhân viên mới", "xin cài lại hệ điều hành",
    ],
    "huong_dan_su_dung": [
        "hỏi cách dùng tính năng báo cáo", "hỏi cách kết nối VPN từ nhà", "hỏi cách in hai mặt",
        "hỏi cách khôi phục file đã xoá", "hỏi cách đổi mật khẩu định kỳ",
    ],
}
URGENCY_MARKERS = {
    "cao": ["gấp, đang chặn công việc", "cần xử lý ngay trong hôm nay", "khẩn cấp, ảnh hưởng cả phòng ban"],
    "trung_binh": ["mong bộ phận IT hỗ trợ sớm", "đã báo cáo hai ngày trước", "cần trước cuối tuần"],
    "thap": ["không gấp, tiện thì xử lý giúp", "hỏi trước để chuẩn bị", "chưa vội, khi nào rảnh"],
}
SENTIMENTS = {
    "tieu_cuc": ["rất bất tiện cho công việc", "đã báo nhiều lần chưa được xử lý", "ảnh hưởng tiến độ nhóm"],
    "trung_tinh": ["nhờ bộ phận IT kiểm tra giúp", "báo để bộ phận nắm thông tin", "xin hỗ trợ khi có thể"],
    "tich_cuc": ["cảm ơn IT đã hỗ trợ nhanh những lần trước", "không vấn đề gì lớn, chỉ báo để biết", "tin tưởng bộ phận IT xử lý tốt"],
}
PRODUCTS = [
    "VPN công ty", "email công ty", "máy in mạng", "laptop công ty", "phần mềm kế toán",
    "wifi văn phòng", "máy chủ nội bộ", "tài khoản domain", "phần mềm CRM", "máy chấm công",
    "phần mềm quản lý kho", "máy scan", "hệ thống họp trực tuyến", "ổ đĩa mạng chung",
    "máy tính bàn phòng kế toán",
]
OPENERS = [
    "Chào bộ phận IT,", "Nhờ IT hỗ trợ,", "Xin chào,", "Kính gửi bộ phận kỹ thuật,", "Chào anh/chị IT,",
]
DEPARTMENTS = ["phòng Kế toán", "phòng Nhân sự", "phòng Kinh doanh", "phòng Marketing", "phòng Vận hành"]
CONNECTORS = ["", " Cụ thể là", " Chi tiết:"]

REQUIRED_KEYS = ["intent", "urgency", "product", "sentiment"]

INSTRUCTION = (
    "Phân loại ticket hỗ trợ CNTT nội bộ sau thành JSON với đúng 4 khóa: "
    "intent, urgency, product, sentiment. Chỉ trả về JSON, không giải thích.\n\n"
    "intent thuộc: loi_phan_mem | loi_phan_cung | cap_quyen | cai_dat_moi | huong_dan_su_dung\n"
    "urgency thuộc: cao | trung_binh | thap\n"
    "sentiment thuộc: tieu_cuc | trung_tinh | tich_cuc\n"
    "product: tên hệ thống/thiết bị xuất hiện trong ticket."
)


def _cap(phrase: str) -> str:
    """Capitalize only the first letter -- str.capitalize() lowercases the rest,
    which mangles the deliberate "IT" acronym in a couple of phrases above."""
    return phrase[0].upper() + phrase[1:]


def make_ticket(rng: random.Random) -> dict:
    intent = rng.choice(list(INTENTS))
    urgency = rng.choice(list(URGENCY_MARKERS))
    sentiment = rng.choice(list(SENTIMENTS))
    product = rng.choice(PRODUCTS)
    dept = rng.choice(DEPARTMENTS)
    connector = rng.choice(CONNECTORS)

    # Randomize clause order so the corpus is not one fixed sentence template.
    clauses = [
        f"Bên {dept} đang dùng {product}, {rng.choice(INTENTS[intent])}.",
        f"{_cap(rng.choice(URGENCY_MARKERS[urgency]))}.",
        f"{_cap(rng.choice(SENTIMENTS[sentiment]))}.",
    ]
    rest = clauses[1:]
    rng.shuffle(rest)
    body = f"{rng.choice(OPENERS)}{connector} {clauses[0]} " + " ".join(rest)
    label = {"intent": intent, "urgency": urgency, "product": product, "sentiment": sentiment}
    return {"ticket": body, "label": label}


def to_record(t: dict) -> dict:
    return {
        "instruction": INSTRUCTION,
        "input": t["ticket"],
        "output": json.dumps(t["label"], ensure_ascii=False),
        "label": t["label"],
    }


def main() -> None:
    rng = random.Random(SEED)
    DATA.mkdir(exist_ok=True)

    seen: set[str] = set()
    tickets: list[dict] = []
    while len(tickets) < 340:
        t = make_ticket(rng)
        if t["ticket"] in seen:
            continue
        seen.add(t["ticket"])
        tickets.append(t)

    train = [to_record(t) for t in tickets[:260]]
    target_eval = [to_record(t) for t in tickets[260:310]]
    holdout = [to_record(t) for t in tickets[310:]]

    train_bodies = {r["input"] for r in train}
    overlap = [r for r in target_eval + holdout if r["input"] in train_bodies]
    assert not overlap, f"CONTAMINATION: {len(overlap)} eval items leaked into train"

    def write(path: pathlib.Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write(DATA / "train_seed.jsonl", train)
    write(DATA / "eval_target.jsonl", target_eval)
    write(DATA / "holdout_secret.jsonl", holdout)

    print(f"train_seed        {len(train):>4}")
    print(f"eval_target       {len(target_eval):>4}")
    print(f"holdout_secret    {len(holdout):>4}")
    print("eval_regression.jsonl left untouched (domain-agnostic general-capability probe)")


if __name__ == "__main__":
    main()
