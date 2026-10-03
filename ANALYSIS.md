# Phân tích kết quả — Day 17: Memory Systems for AI Agent

Số liệu lấy từ `python src/benchmark.py` (offline, deterministic). Token được ước lượng bằng `len(text)/4`.

## 1. Kết quả

**Standard Benchmark** — 10 hội thoại, 101 lượt, user `dungct`

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 1,886 | 12,886 | 0.00 | 0.00 | 0 | 0 |
| Advanced | 2,400 | 21,719 | 1.00 | 1.00 | 380 | 0 |

**Long-Context Stress Benchmark** — 1 hội thoại 16 lượt rất dài, user `dungct_stress`

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2,526 | 21,806 | 0.00 | 0.00 | 0 | 0 |
| Advanced | 2,650 | 11,797 | 1.00 | 1.00 | 237 | 3 |

Câu chuyện đi theo đúng thứ tự của rubric:

1. Baseline không nhớ dài hạn: recall = 0 vì mỗi recall question chạy ở thread mới.
2. Advanced thêm `User.md` nên recall tăng lên 1.00 (kể cả với correction Huế → Đà Nẵng và nhiễu "Hà Nội", "product manager").
3. Hội thoại dài làm prompt cost của baseline tăng mạnh: mỗi lượt gửi lại toàn bộ lịch sử, nên tổng chi phí tăng gần bậc hai theo số lượt.
4. Compact memory kéo chi phí ngữ cảnh xuống: Advanced xử lý **ít hơn 46%** prompt tokens trong stress test.
5. Hệ thống mạnh hơn nhưng phức tạp hơn và cần guardrail (mục 4).

## 2. Vì sao compact không phải lúc nào cũng thắng

Ở bộ Standard, Advanced xử lý **nhiều hơn 69%** prompt tokens và tốn nhiều hơn 27% agent tokens:

- Mỗi hội thoại chỉ ~10 lượt ngắn (~20 token/lượt), chưa bao giờ chạm ngưỡng compact nên `Compactions = 0`: không có gì để tiết kiệm.
- Trong khi đó `User.md` (system prompt + hồ sơ) được nạp vào **mọi** lượt. Với lịch sử ngắn, phần cố định này lớn hơn chính lịch sử mà baseline phải mang theo.
- `Agent tokens only` cao hơn vì Advanced trả lời lượt "ghi nhớ" bằng câu xác nhận dài hơn và phải trả token cho lệnh ghi memory (mô phỏng bằng 8 token overhead + payload mỗi lần `User.md` thay đổi).

Memory chỉ có lợi khi chi phí cố định của nó được bù bằng recall (luôn có) và bằng tiết kiệm ngữ cảnh (chỉ có khi thread đủ dài).

## 3. Vì sao compact chủ yếu tối ưu `Prompt tokens processed`

Trong stress test, `Agent tokens only` gần như không đổi (+5%): đó là số token của chính cuộc trò chuyện (tin người dùng + câu trả lời), compact không làm chúng nhỏ đi. Thứ compact thay đổi là lượng ngữ cảnh phải **đọc lại** ở mỗi lượt: baseline mang cả lịch sử (tăng tuyến tính theo lượt), advanced mang `User.md` + summary có giới hạn + vài tin gần nhất (bị chặn trên bởi ngưỡng ~1000 token). Vì vậy lợi ích nằm ở cột prompt, và lớn dần khi thread dài thêm.

## 4. Ba lớp memory và rủi ro đi kèm

| Lớp | Lưu ở đâu | Sống bao lâu | Rủi ro chính |
|---|---|---|---|
| Short-term | `messages` gần nhất của thread | trong thread | Tràn ngữ cảnh → tốn token; baseline chỉ có lớp này |
| Persistent | `User.md` | qua mọi thread | Lưu sai fact; file phình; dữ liệu cá nhân nằm dạng plain text |
| Compact | `summary` của thread | trong thread | Summary mất chi tiết (bản offline là heuristic cắt câu đầu) |

- **File phình to**: tổng cộng 380 byte cho 101 lượt. Được giữ nhỏ nhờ chỉ lưu fact ổn định, fact đơn trị bị thay thế thay vì cộng dồn, và fact đa trị bị chặn 6–8 mục (cũ nhất rụng trước). Không có guardrail này thì mỗi lượt đều có thể thêm dòng mới.
- **Lưu sai fact**: rủi ro lớn nhất vì sai một lần là sai ở mọi thread sau. Các nguồn sai trong dữ liệu: câu hỏi ("Tên mình là gì?" bị đọc thành tên "gì"), câu đùa ("chuyển sang product manager"), nơi đi họp ("Hà Nội"), vế cũ của correction ("Lúc đầu mình nói ở Huế"), và `giải thích` bị đọc nhầm là `thích`. Mỗi trường hợp đều có test riêng.
- **Summary mất thông tin**: chấp nhận được vì fact bền vững được rút ra ngay khi người dùng nói, không phụ thuộc summary. Test `test_facts_survive_compaction` kiểm tra điều này.
- **An toàn dữ liệu**: `User.md` không mã hóa; user id được làm sạch khi tạo đường dẫn (chặn `../`); tool ghi memory ở live mode chỉ nhận field trong whitelist và giá trị tối đa 120 ký tự để hạn chế bị lợi dụng ghi nội dung tùy ý.

## 5. Phần bonus (rubric 90–100)

| Bonus | Giải quyết vấn đề gì | Cải thiện gì | Rủi ro tạo thêm |
|---|---|---|---|
| **Confidence threshold** (`MIN_CONFIDENCE=0.7`) | Không ghi tín hiệu yếu như thói quen "mình vẫn uống cà phê sữa đá" như thể là sở thích | Tránh ghi sai; giữ file nhỏ. Fact vẫn được lưu khi người dùng nêu rõ ("đồ uống yêu thích là...") | Bỏ sót fact nếu người dùng chỉ nói gián tiếp; ngưỡng là con số chỉnh tay |
| **Conflict handling** | Correction phải thay fact cũ, không giữ song song hai giá trị mâu thuẫn | Recall đúng sau correction (Huế, MLOps engineer, Đà Nẵng); giá trị kém cụ thể ("MLOps") không ghi đè giá trị cụ thể ("MLOps engineer") | Mất lịch sử (không biết trước đó ở đâu); một correction nhận diện sai sẽ xóa fact đúng |
| **Entity extraction** | Fact có cấu trúc (`name`, `location`, `profession`, `pet`, `style`...) thay vì chuỗi tự do | Câu trả lời recall chính xác theo từng field; `style` được chuẩn hóa ("bullet" không ghi đè "3 bullet") | Schema cố định, ngoài schema thì không nhớ |
| **Memory decay (bản nhẹ)** | Danh sách sở thích/style không được tăng vô hạn | Mục lâu không nhắc tự rụng; mục nhắc lại được đẩy lên cuối | Có thể rụng mất một fact thật sự quan trọng nhưng ít nhắc |

## 6. Giới hạn cần đọc kèm kết quả

- **Recall 1.00 không phải ước lượng hiệu năng thực tế.** Extractor là rule-based và được viết, kiểm chứng trên chính bộ dữ liệu này; nó dễ vỡ với cách diễn đạt khác. Con số cho thấy kiến trúc memory chạy đúng, không cho thấy khả năng khái quát hóa. Live mode (LLM tự gọi tool ghi memory, kèm extractor làm guardrail) là hướng để vượt giới hạn này, nhưng chưa được đo với model thật trong repo này vì không có API key.
- Baseline cố ý dùng cùng extractor để hỏi công bằng: nó nhớ được trong cùng thread nhưng không sang thread mới.
- `Response quality` ở offline là heuristic (recall nhân hệ số ngắn gọn) nên gần như trùng recall; ở `--live` có thể dùng judge model.
- Token là ước lượng `len/4`, chỉ có ý nghĩa so sánh tương đối. Chi phí các câu hỏi recall không được tính vào hai cột token.
- Stress test chỉ có 1 hội thoại và 3 lần compact; kết quả −46% phụ thuộc ngưỡng (`COMPACT_THRESHOLD_TOKENS=1000`) và `COMPACT_KEEP_MESSAGES=4`.
