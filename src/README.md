# `src/` — Memory Systems for AI Agent (bản hoàn thiện)

Toàn bộ code chạy được **offline, không cần API key**. Live mode (LangChain/LangGraph) là phần mở rộng
tùy chọn và tự tắt nếu không có key hợp lệ.

```bash
python src/benchmark.py                 # Standard + Long-Context Stress (offline, lặp lại được)
python src/benchmark.py --suite stress  # chỉ chạy một bộ
python src/benchmark.py --live          # dùng LLM thật nếu có key; không có key thì tự chạy offline
pytest src/test_agents.py -v
```

## Các file

| File | Vai trò |
|---|---|
| `model_provider.py` | `ProviderConfig`, `normalize_provider()` (alias như `anthorpic`), `build_chat_model()` cho 6 provider (import lười), `has_live_credentials()` (bỏ qua placeholder như `...`) |
| `config.py` | `LabConfig`, `load_config()`: đọc env + `.env`, tự chọn provider theo key, tạo `state/` |
| `memory_store.py` | `estimate_tokens()`, `UserProfileStore` (`User.md`), `extract_profile_updates()`, `summarize_messages()`, `CompactMemoryManager` |
| `agent_baseline.py` | Agent A: chỉ nhớ trong cùng thread |
| `agent_advanced.py` | Agent B: short-term + `User.md` + compact memory |
| `benchmark.py` | Hai bộ benchmark, 6 cột chỉ số, bảng markdown |
| `test_agents.py` | 50 test: `User.md`, extractor, compact, cross-session recall, prompt load, benchmark, provider/config, live fallback |

## Offline và live

- `force_offline=True` (benchmark/test mặc định): luôn deterministic.
- `force_offline=False`: dùng live nếu có key hợp lệ **và** cài `langchain`; ngược lại offline.
- Nếu một lượt live lỗi (mạng, auth, quota...), agent cảnh báo một lần rồi chạy tiếp offline với đầy đủ history, không raise.

Baseline và Advanced dùng **cùng một extractor và cùng bộ soạn câu trả lời**; khác biệt duy nhất là kiến trúc memory
(baseline chỉ nhớ message của thread hiện tại, advanced đọc `User.md`), nên benchmark đo đúng memory chứ không đo "độ thông minh".

## Biến môi trường (tùy chọn, đặt trong `.env`)

```
LLM_PROVIDER=openai|custom|gemini|anthropic|ollama|openrouter   # bỏ trống: tự chọn theo key có sẵn
LLM_MODEL=...            LLM_TEMPERATURE=0
OPENAI_API_KEY  GEMINI_API_KEY (hoặc GOOGLE_API_KEY)  ANTHROPIC_API_KEY  OPENROUTER_API_KEY
CUSTOM_BASE_URL  CUSTOM_API_KEY  OLLAMA_BASE_URL
JUDGE_PROVIDER  JUDGE_MODEL                       # model chấm điểm khi chạy --live
COMPACT_THRESHOLD_TOKENS=1000  COMPACT_KEEP_MESSAGES=4
```

## Quyết định thiết kế đáng biết

- `User.md` là các dòng `- key: value` dễ đọc, dễ sửa tay; chỉ ghi lại các dòng bị đổi, giữ nguyên ghi chú viết tay.
- Fact đơn trị (`name`, `location`, `profession`, ...) bị **thay thế** khi có correction; fact đa trị (`interests`, `hobbies`, `style`) có giới hạn số mục và ưu tiên mục mới nhắc.
- Extractor bỏ qua câu hỏi, câu đùa/giả định, vế "cũ" của một correction, và các tín hiệu yếu dưới ngưỡng tin cậy (`MIN_CONFIDENCE = 0.7`).
- Benchmark mỗi bộ dùng `state/benchmark/<suite>/` được xóa sạch trước khi chạy; chi phí token chỉ tính trên các thread hội thoại, không tính các câu recall.
- Phân tích kết quả: xem `../ANALYSIS.md`.
