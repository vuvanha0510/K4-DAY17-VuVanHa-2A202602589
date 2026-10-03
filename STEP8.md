# Bước 8. Phân tích kết quả benchmark

Tài liệu này **không** mô tả lại tính năng của hệ thống. Mỗi luận điểm dưới đây được viết theo cùng một công thức:

> **số liệu nào** (bảng benchmark + cột cụ thể) → **cơ chế nào trong `src/` tạo ra số đó** (file + hàm) → **giới hạn nào đi kèm**.

Mọi con số dưới đây lấy từ hai lệnh sau, chạy offline hoàn toàn (`force_offline=True`), không cần API key:

```bash
python src/benchmark.py          # in hai bảng
pytest src/test_agents.py -v     # 4 passed
```

---

## 0. Hai bảng đo được

### Standard Benchmark — `data/conversations.json` (10 hội thoại, 101 lượt, 14 câu hỏi recall, user `dungct`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 3,497 | 24,526 | 0.10 | 0.37 | 0 | 0 |
| Advanced | 10,192 | 53,371 | **0.96** | 0.97 | 352 | 82 |
| *Advanced − Baseline* | *+6,695* | *+28,845* | *+0.86* | *+0.60* | *+352* | *+82* |

### Long-Context Stress Benchmark — `data/advanced_long_context.json` (1 hội thoại, 16 lượt, 3 câu hỏi recall)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 720 | 26,999 | 0.00 | 0.30 | 0 | 0 |
| Advanced | 971 | **16,548** | **1.00** | 1.00 | 294 | 26 |
| *Advanced − Baseline* | *+251* | ***−10,451 (−38.7%)*** | *+1.00* | *+0.70* | *+294* | *+26* |

Bốn mắt xích mà reviewer phải thấy — baseline không nhớ dài hạn → advanced có `User.md` nên recall tăng → hội thoại dài làm prompt cost tăng mạnh → compact kéo chi phí ngữ cảnh xuống → hệ thống mạnh hơn nhưng phức tạp hơn — được chứng minh từng mắt bằng hai bảng trên.

---

## 1. Vì sao Advanced có recall tốt hơn Baseline

**Số liệu.** Ở cả hai bảng, cột `Cross-session recall` của Advanced đều cao hơn Baseline:

| Bảng | Baseline | Advanced |
|---|---|---|
| Standard (14 câu hỏi) | 0.10 | **0.96** |
| Stress (3 câu hỏi) | 0.00 | **1.00** |

Điểm cần nói rõ: **con số 0.10 của Baseline ở bảng Standard không phải là nhớ được gì cả.** Nó là hiệu ứng nhiễu của chính cách đo. `benchmark.run_agent_benchmark()` hỏi recall trong thread mới `<conv>::recall`, nhưng một thread recall dùng chung cho *các câu hỏi của cùng một hội thoại*. `BaselineAgent._search_thread()` (`src/agent_baseline.py:116`) quét lại các message **người dùng** đã có trong thread đó — mà message đó chính là các câu hỏi recall trước. Với câu hỏi *"Bạn biết DũngCT là ai không?"*, Baseline trả lời:

```
Trong thread này mình nhớ: - Bạn biết DũngCT là ai không?
```

Nó **echo lại câu hỏi** nên `recall_points()` khớp được chuỗi `DũngCT` và cho 0.667 cho câu đó. Ba câu hỏi duy nhất bị vậy (conv-05, conv-06, conv-09) kéo mức trung bình lên 0.10. Ở bảng Stress, ba câu hỏi đều không tự chứa từ khóa trong `expected_contains`, nên Baseline rơi về đúng **0.00** — đây mới là con số trung thực cho "baseline không nhớ dài hạn".

**Cơ chế.** Đường đi của một fact qua ba hàm:

1. `extract_profile_updates(message)` (`src/memory_store.py:510`) quét message bằng regex, chỉ nhận các mẫu câu *khẳng định* — `_LOCATION_NOISE` loại câu kiểu "Hà Nội chỉ là nơi đi họp", `_DRINK_INTENT` bắt buộc có ý "thích/yêu thích" trước khi lưu đồ uống, và `is_question_only()` loại câu hỏi.
2. `AdvancedAgent._reply_offline()` (`src/agent_advanced.py:104-107`) ghi fact vào `User.md` qua `UserProfileStore.upsert_fact()`, với `mode="replace"` cho `SINGLE_VALUE_FACTS` (name, location, profession…) và `mode="union"` cho sở thích/style.
3. `_offline_response()` (`src/agent_advanced.py:156`) **không** đọc lịch sử thread. Nó gọi `self.profile_store.facts(user_id)` — parse trực tiếp từ file — rồi `_select_facts()` dò câu hỏi theo `_FACT_ROUTING` và trả về đúng các fact liên quan. Vì vậy câu hỏi ở thread mới vẫn có đáp án.

**Correction thật, không phải đoán may.** Ở bảng Stress, lượt 9 nói *"Lúc đầu mình nói hiện ở Huế, nhưng thực ra từ tuần này mình đang làm việc ở Đà Nẵng"*. Lượt 10 thêm nhiễu (*"hay là chuyển sang product manager cho đỡ phải ngồi cứng"*), lượt 11–13 thêm tin tức dài. Kết quả trong `state/profiles/dungct-stress/User.md`:

```
- name: DũngCT Stress
- location: Đà Nẵng              ← fact mới thắng, Huế bị replace
- profession: MLOps engineer     ← "product manager" bị loại
- response_style: 3 bullet; trả lời ngắn gọn; ưu tiên nhấn trade-off; có ví dụ thực chiến; trả lời dạng bullet
```

Câu hỏi recall thứ hai cố hỏi *"Nếu ai đó nhắc Huế, Hà Nội hay product manager, đâu mới là nghề nghiệp và nơi ở hiện tại?"* và Advanced trả lời đúng `MLOps engineer` + `Đà Nẵng` (1.00). Ở bảng Standard, `User.md` cũng chứa đúng `location: Huế` (sau câu "không còn ở Đà Nẵng") và `profession: MLOps engineer` (sau câu "không còn làm backend engineer nữa").

**Giới hạn.** Advanced **không** đạt 1.00 ở bảng Standard: câu *"Hiện tại mình làm nghề gì và mình còn ở Huế không?"* chỉ được 0.50. Nguyên nhân nằm ở `_FACT_ROUTING` (`src/agent_advanced.py:26`): từ khoá của `location` là `"ở đâu", "nơi ở", "đang ở", "sống ở", "hiện ở"`, mà câu hỏi dùng cụm *"còn ở Huế"* — không trigger nào khớp, nên `_select_facts()` bỏ qua `location` dù fact đã có trong file. Đây là giới hạn của **router theo từ khoá**, không phải của lớp trích xuất fact.

---

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn

**Số liệu (bảng Standard).**

| Cột | Baseline | Advanced | Chênh |
|---|---|---|---|
| `Agent tokens only` | 3,497 | 10,192 | +6,695 (×2.91) |
| `Prompt tokens processed` | 24,526 | 53,371 | +28,845 (×2.18) |

**Cơ chế (ba nguồn chi phí, tất cả đều nằm trong code):**

1. **Mang profile đi theo mỗi lượt.** `_estimate_prompt_context_tokens()` (`src/agent_advanced.py:138`) cộng `profile_store.read_text(user_id)` vào mọi lượt. `User.md` sau 101 lượt dài 352 byte ≈ **86 token**; nhân với 115 lượt (101 turn + 14 câu hỏi recall) là **≈ 9.890 token**, tức ~34% trong tổng chênh lệch 28.845. Baseline không có dòng này — `_render(session)` (`src/agent_baseline.py:152`) chỉ join message của thread.
2. **Câu trả lời dài hơn.** `_offline_response()` liệt kê *mọi* fact đọc được từ `User.md` khi câu hỏi không trigger rõ. Trung bình Advanced sinh **100.9 token/lượt** so với **34.6** của Baseline (dòng fallback ngắn của `_offline_response()` ở `agent_baseline.py:109`). Nhân với 101 lượt ra đúng ~6.697 — khớp cột `Agent tokens only` chênh +6.695.
3. **Ghi file.** `upsert_fact()` chạy mỗi lượt, kéo theo `write_text()` render lại toàn bộ `User.md`. Ở bảng Standard có **15 lần ghi / 3.351 byte ghi ra** cho một file cuối cùng chỉ 352 byte. (Không tính vào token, nhưng là chi phí I/O thật.)

**Vì sao compact không bù được ở đây.** Đây là phần quan trọng nhất của luận điểm. Ở bảng Standard, thread chỉ dài ~10 lượt và nội dung ngắn, nên **Baseline vốn đã chưa tốn nhiều ngữ cảnh**: thread đầy đủ của nó chỉ 463–777 token (đo bằng `estimate_tokens(base._render(...))`). Trong khi đó ngưỡng compact mặc định là `COMPACT_THRESHOLD_TOKENS=600` và giữ `COMPACT_KEEP_MESSAGES=6` (`src/config.py:148`). Hệ quả đo được: context sống của Advanced sau khi compact (521–1.102 token/thread) **lớn hơn cả** thread đầy đủ của Baseline. Compact đã cắt (82 lần, ~8.2 lần/thread) nhưng sàn của nó — summary + 6 message gần nhất + `User.md` — vẫn cao hơn chi phí của Baseline. Advanced thắng ở recall, thua ở token: đó chính là trade-off của hội thoại ngắn.

**Giới hạn.** Ngưỡng 600 token / keep 6 message được chọn để stress test; với hội thoại 10 lượt nó tốn kém. Hệ thống hiện chưa có ngưỡng thích ứng theo độ dài thread, và cũng chưa có memory decay để thu nhỏ `User.md` theo thời gian.

---

## 3. Vì sao compact có lợi thế ở hội thoại dài

**Số liệu (bảng Stress).** Điểm mấu chốt nằm ở **cột `Prompt tokens processed`**, không phải `Agent tokens only`:

| Cột | Baseline | Advanced | Chênh |
|---|---|---|---|
| `Prompt tokens processed` | 26,999 | **16,548** | **−10.451 (−38.7%)** |
| `Agent tokens only` | 720 | 971 | +251 (+34.9%) |

Hai cột này **nói ngược chiều nhau**, và đó chính là bản chất của compact memory:

- **Compact tối ưu cột ngữ cảnh, không tối ưu cột sinh token.** `CompactMemoryManager._maybe_compact()` (`src/memory_store.py:674`) giữ 6 message gần nhất, đưa phần cũ vào `state["summary"]`; `render()` chỉ trả về summary + message đang giữ. Nhưng `_offline_response()` vẫn sinh câu trả lời **dài hơn** Baseline (971 > 720), vì nó liệt kê profile. Viết câu "compact giúp tiết kiệm token" mà không tách hai cột là sai: compact **không** tiết kiệm `Agent tokens only`, nó tiết kiệm `Prompt tokens processed`.
- **Số liệu per-turn cho thấy tại sao.** Chi phí ngữ cảnh mỗi lượt của Baseline trong stress tăng **tuyến tính, không chặn trên**: lượt 2 = 241 token → lượt 5 = 942 → lượt 10 = 2.041 → lượt 16 = **3.279 token/lượt**. Đó là hệ quả trực tiếp của việc `_reply_offline()` của Baseline cộng `estimate_tokens(self._render(session))` — cả thread, mỗi lượt (`agent_baseline.py:74-76`) — và `compaction_count()` của nó **luôn trả 0** (`agent_baseline.py:58`).
- Advanced nhìn ngược lại: lượt 2 = 513, lượt 5 = 802, lượt 10 = 1.126, lượt 16 = **1.607 token/lượt** — tăng chậm, và **bị chặn trên** bởi ngưỡng 600 + keep 6. 26 lần compact xảy ra ở các lượt 4, 5, 6, 7, 8, …, 16, tức gần như **mỗi 2 lượt một lần** khi ngữ cảnh đã dài.
- Bằng chứng cấu trúc: thread stress có **32 message**, nhưng `context()` chỉ còn **6 message** + 1 summary → **26 message đã bị nén**.

**Giới hạn (đây là cái giá của compact).** `summarize_messages()` (`src/memory_store.py:566`) chỉ giữ **câu đầu tiên** của mỗi message, cắt còn 140 ký tự, tối đa 6 message mỗi lần nén. Đây là nén **lossy**: chi tiết tin tức ở các lượt 6, 11, 12, 15 không còn trong context sống. Hệ quả quan sát được lúc chạy: **summary vẫn giữ dấu vết của fact cũ** — trong `context()["summary"]` của thread stress có câu *"Nơi ở hiện tại: Huế"* từ trước khi correction, trong khi `User.md` đã là `Đà Nẵng`. Hệ thống hiện phải chịu mâu thuẫn nội bộ giữa lớp persistent và lớp compact; một thiết kế tốt hơn là khi có correction thì phải ghi đè/invalidate phần summary liên quan.

## 4. File memory tăng trưởng ra sao và rủi ro gì

**Số liệu.**

| Bảng | `Memory growth (bytes)` Advanced | `Memory growth (bytes)` Baseline | `Compactions` Advanced | `Compactions` Baseline |
|---|---|---|---|---|
| Standard | 352 | **0** | 82 | **0** |
| Stress | 294 | **0** | 26 | **0** |

Hai điều reviewer nhìn thấy ngay:

- **Baseline bằng 0 byte ở cả hai bảng** — không phải vì file nhỏ, mà vì nó không ghi file nào: `run_agent_benchmark()` chỉ cộng `memory_bytes` qua `getattr(agent, "memory_file_size", None)`, và `BaselineAgent` không có phương thức này. Cùng một cơ chế đó giải thích `Compactions = 0` (`agent_baseline.py:58`).
- **File Advanced phình *có kiểm soát*, không phình tuyến tính.** Sau 101 lượt (10 thread, 11 fact) file là 352 byte; sau 16 lượt stress (5 fact) là 294 byte. Ba cơ chế trong `UserProfileStore` giữ nó lại:
  - `render_facts()` (`memory_store.py:138`) chỉ ghi các key có trong `FACT_ORDER` cộng key mới, vào template cố định → **không bao giờ append tự do**.
  - `upsert_fact(mode="replace")` cho fact một giá trị → fact sửa là **ghi đè**, không cộng dồn (Đà Nẵng thay Huế, MLOps thay backend engineer).
  - `upsert_fact(mode="union", max_items=12)` giữ tối đa 12 phần tử và loại trùng lặp. Trong stress, `response_style` dài nhất là 5 mục, không phình vô hạn dù người dùng lặp lại sở thích.
- **Nhưng chi phí ghi không nằm ở kích thước file mà ở số lần ghi.** Vì `write_text()` render lại toàn file, Standard ghi **15 lần / 3.351 byte** cho file cuối 352 byte (~9.5×), stress ghi **9 lần / 1.723 byte** cho 294 byte (~5.9×). Cột `Memory growth (bytes)` đang đo kích thước **tức thời**, không đo **lưu lượng ghi** — đây là hạn chế của chính phép đo, cần nói ra.

**Rủi ro cụ thể quan sát được trong lúc chạy:**

1. **File phình theo thời gian và là chi phí cố định mỗi lượt.** `User.md` 86 token bị nhân lên ở **mọi** lượt qua `_estimate_prompt_context_tokens()`, kể cả lượt không liên quan gì đến profile. Thread càng dài, phần này càng ít ý nghĩa so với summary. Biện pháp là `Memory decay`.
2. **Fact sai bị giữ lại sau một lượt nhiễu.** Regex `extract_profile_updates()` lo được nhiễu *đã biết trước* ("product manager" là câu đùa, "Hà Nội" chỉ là nơi đi họp), nhưng **không có confidence threshold** — một câu nói đùa lạ chưa từng thấy vẫn có thể vào `User.md` và được trả lời như sự thật ở thread sau. Với `mode="union"` còn tệ hơn: giá trị sai được **giữ lại cạnh** giá trị đúng thay vì thay thế nó. Nguyên lý hiện tại của `upsert_fact()` chỉ quyết định "replace hay union" theo *loại fact*, chứ không theo *độ tin cậy của câu* — đó chính là chỗ cần `Confidence threshold`.
3. **Compact có thể làm *thu hẹp* thông tin quan trọng hơn mức chấp nhận được.** `keep_messages=6` + cắt 140 ký tự: ở stress, 26/32 message biến mất khỏi context sống.
4. **Hai lớp memory mâu thuẫn với nhau.** Như đã nêu ở Mục 3: summary giữ "Huế" (trước correction) trong khi `User.md` đã là "Đà Nẵng". Chỉ `_offline_response()` ưu tiên `User.md` nên câu trả lời mới đúng; bất kỳ đường đi nào đọc summary trước file sẽ sai.

---

## 5. Kết luận: mạnh hơn, nhưng phức tạp hơn và cần guardrail tốt hơn

Bảng số nói rõ ranh giới giữa hai lớp chi phí:

| | Standard (101 lượt, ngắn) | Stress (16 lượt, rất dài) |
|---|---|---|
| `Prompt tokens processed` | Advanced **tệ hơn** ×2.18 | Advanced **tốt hơn** −38.7% |
| `Agent tokens only` | Advanced tệ hơn ×2.91 | Advanced tệ hơn +34.9% |
| `Cross-session recall` | 0.10 → **0.96** | 0.00 → **1.00** |

Ba kết luận rút ra từ chính các con số này:

1. **Đổi một cột (recall) lấy hai cột (token).** Advanced mua recall bằng cách trả tiền ở `Agent tokens only` và ở chi phí ngữ cảnh cố định của `User.md`. Ở hội thoại ngắn, giao dịch đó **lỗ**; ở hội thoại dài, nhờ compact, nó **thắng** — nhưng chỉ thắng ở cột `Prompt tokens processed`.
2. **Guardrail hiện có là "lọc bằng regex", chưa phải guardrail.** `memory growth` đã bị chặn bởi template + `replace`/`union` + `max_items`, và correction thắng được fact cũ. Nhưng chưa có confidence threshold, chưa có decay, chưa có xử lý xung đột giữa summary và `User.md`. Ba thứ đó là đúng ba hướng bonus của Bước 9.
3. **Phép đo còn thiếu chỉ số.** `Memory growth (bytes)` chỉ chụp kích thước file tức thời, không chụp số lần ghi (15 lần / 3.351 byte so với 352 byte ở Standard) và không chụp được fact sai vẫn còn sống. Thêm hai chỉ số như `Profile writes` và một tập câu hỏi recall **cố tình đặt nhiễu chưa từng thấy trong dataset** sẽ làm lộ rủi ro thật, thay vì chỉ cho thấy hệ thống chạy đúng.

---

Xác minh lại toàn bộ số trong tài liệu này bằng:

```bash
python src/benchmark.py          # hai bảng ở Mục 0
pytest src/test_agents.py -v     # 4 passed: User.md, compact trigger, cross-session recall, prompt load giảm
```
---

