# Tài Liệu Quy Trình Hệ Thống (Workflow) & Tối Ưu Hóa HybridACD

Tài liệu này tổng hợp toàn diện cơ sở lý thuyết nghiên cứu ban đầu của nhóm tác giả, các điều kiện thực thi dự án, đề xuất chi tiết về phương pháp tối ưu hóa **HybridACD (Hybrid Adversarial Consistency Defense)**, danh sách các mô hình AI sử dụng qua API và quy trình các bước chạy dự án cụ thể.

---

## 1. Tổng Quan Nghiên Cứu Ban Đầu Của Nhóm Tác Giả

Nghiên cứu gốc *"Consistency Checks for Language Model Forecasters"* (ICLR 2025) đề xuất phương pháp đánh giá năng lực dự báo của các Mô hình Ngôn ngữ Lớn (LLM) thông qua **Tính nhất quán logic nội tại (logical consistency)** thay vì phụ thuộc hoàn toàn vào kết quả thực tế ở tương lai (ground truth), giúp giải quyết bài toán "Scalable Oversight" và tránh rò rỉ dữ liệu huấn luyện (data contamination).

### A. Nguyên lý Toán học & Định lý Dutch Book
*   **Arbitrage Metric (Độ vi phạm chênh lệch giá):** Vay mượn từ lý thuyết kinh tế học hành vi. Nếu các xác suất dự báo do LLM đưa ra vi phạm các quy tắc logic cơ bản, một nhà đầu tư có thể thiết lập các giao dịch mua/bán chéo để rút tiền chắc chắn từ mô hình (gọi là "cỗ máy bào tiền" - money pump).
*   **LMSR (Logarithmic Market Scoring Rules):** Tích hợp hàm lô-ga-rít tự nhiên để trừng phạt khắc nghiệt các lỗi vi phạm có tính tự tin thái quá (overconfident violations) ở khu vực cận biên (gần 0 và 1).
*   **Frequentist Metric (Độ vi phạm tần suất):** Sử dụng các kiểm định thống kê và mô phỏng Monte Carlo để định lượng xem sai lệch của LLM là do nhiễu ngẫu nhiên hay do sự sụp đổ thực sự trong hệ thống niềm tin logic của mô hình.

### B. 10 Bộ Kiểm Tra Logic Nhất Quán (Consistency Checkers)
Hệ thống sử dụng LLM để xây dựng các bộ câu hỏi liên đới logic (Question Tuples) nhằm kiểm toán mô hình trên 10 quy tắc toán học và xác suất:
1.  **Negation (Phủ định):** $P(P) + P(\neg P) = 1$
2.  **Paraphrase (Cách diễn đạt khác):** $P(P) = P(Q)$
3.  **Consequence (Hệ quả):** Nếu $P \implies Q$ thì $P(P) \le P(Q)$
4.  **AndOr:** $P(P) + P(Q) = P(P \lor Q) + P(P \land Q)$
5.  **And:** $\max(P(P) + P(Q) - 1, 0) \le P(P \land Q) \le \min(P(P), P(Q))$
6.  **Or:** $\max(P(P), P(Q)) \le P(P \lor Q) \le \min(1, P(P) + P(Q))$
7.  **But:** $P(P \lor Q) = P(P) + P(\neg P \land Q)$
8.  **Conditional (Xác suất có điều kiện):** $P(P) \cdot P(Q \vert P) = P(P \land Q)$
9.  **CondCond (Xác suất có điều kiện lồng nhau):** $P(P) \cdot P(Q \vert P) \cdot P(R \vert P \land Q) = P(P \land Q \land R)$
10. **Expected Evidence (Kỳ vọng minh chứng):** $P(P) = P(P \vert Q)P(Q) + P(P \vert \neg Q)(1 - P(Q))$

---

## 2. Điều Kiện Để Thực Hiện Dự Án

### A. Điều kiện chính (Primary Conditions)
*   **Môi trường thực thi:** Python 3.11 (được khuyến nghị chính thức) hoặc Python 3.14 (sử dụng các gói thư viện cài đặt toàn cục).
*   **Tệp phụ thuộc:** Đã được cài đặt đầy đủ thông qua tệp [requirements.txt](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/requirements.txt).
*   **Cấu hình biến môi trường:** Tệp [.env](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/.env) phải chứa đầy đủ API key và proxy hoạt động cho các mô hình AI (OpenAI, Anthropic, Gemini, Qwen, Perplexity).
*   **Dữ liệu thực nghiệm:** Các bộ dữ liệu câu hỏi dự báo (`src/data/fq/real/`, `src/data/fq/synthetic/`) và các bộ dữ liệu tuples logic (`src/data/tuples/scraped/`, `src/data/tuples/newsapi/`, `src/data/tuples/2028/`) phải có sẵn trong thư mục làm việc.

### B. Điều kiện phụ (Secondary Conditions)
*   **Công cụ gán nhãn Streamlit:** Cần chạy trên môi trường ảo cô lập (`.venv_labeling`) hoặc thông qua `pipx` để tránh xung đột thư viện `streamlit` với các phụ thuộc chính của dự án.
*   **Phân tích và Trực quan hóa:** Môi trường hỗ trợ chạy file Jupyter Notebook ([results_analysis.ipynb](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/results_analysis.ipynb)) để tính toán tương quan Pearson và vẽ biểu đồ phân phối.
*   **Thư viện Tokenizer (`tiktoken`):** Cần thiết để mã hóa token và tính toán chính xác Logit Bias khi can thiệp vào quá trình sinh kết quả của LLM.

---

## 3. Đề Xuất Phương Pháp Tối Ưu Hóa HybridACD

Phương pháp **HybridACD (Hybrid Adversarial Consistency Defense)** là giải pháp thế hệ thứ ba kết hợp cơ chế phòng thủ đối kháng động ở đầu vào và can thiệp giải mã toán học ở đầu ra. Phương pháp này giải quyết triệt để sự đánh đổi giữa chi phí tính toán đắt đỏ của thuật toán đệ quy hậu kỳ (`ArbitrageForecaster` tiêu tốn ~$2,500 USD/câu hỏi) và rủi ro sụp đổ năng lực do luật Goodhart của việc huấn luyện tinh chỉnh trực tiếp (`Training for Consistency` khiến AI có xu hướng luôn đoán 50%).

### A. Phương thức hoạt động (Core Mechanisms)
1.  **Adversarial Input Agent (Tác tử đối kháng tiền xử lý):** Sử dụng một mô hình LLM phụ (ví dụ: `Llama-3-70B`) để viết lại câu hỏi gốc dưới dạng phức tạp về ngữ pháp và từ vựng (thêm nhiễu cú pháp, hoán đổi thực thể, lồng ghép logic điều kiện phức tạp...) nhưng bảo toàn hoàn chỉnh ý nghĩa toán học. Điều này buộc mô hình Forecaster phải suy luận logic nhân quả sâu thay vì dựa vào các phản xạ grammar đơn giản.
2.  **Chain-of-Thought (CoT):** Kích hoạt không gian lập luận tự do (scratchpad) cho Forecaster chính để mô hình dàn phẳng các nhiễu loạn từ vựng và tự điều chỉnh phân phối xác suất.
3.  **Token Constraint Decoding (TCD - Ràng buộc giải mã token):** 
    *   Hệ thống theo dõi các dự báo xác suất của các câu hỏi trước đó trong cùng một Tuple logic.
    *   Tính toán khoảng xác suất hợp lệ toán học `[lower_bound, upper_bound]` cho câu hỏi tiếp theo dựa trên 10 quy tắc logic của Checker tương ứng.
    *   Can thiệp trực tiếp vào logits của mô hình phân tích (`gpt-5.4-mini`) ở thời điểm giải mã (decoding-time) thông qua cơ chế `logit_bias`. Mọi token số nằm ngoài khoảng `[lower_bound, upper_bound]` sẽ bị chặn (gán bias âm vô cực), ép buộc xác suất đầu ra phải nằm trong vùng nhất quán logic.

### B. Cách thực hiện chi tiết trong Code
Phương pháp này được cài đặt trực tiếp trong lớp [HybridACDForecaster](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/src/forecasters/hybrid_acd_forecaster.py#L28) kế thừa từ lớp cha `Forecaster`:
*   **Adversarial Rewrite:** Thực hiện qua hàm `adversarial_rewrite_sync` / `adversarial_rewrite_async` gửi prompt đối kháng đến `adversarial_model`.
*   **Consistency Bounds Calculation:** Hàm `get_consistency_bounds` lập bản đồ điều kiện biên cho toàn bộ 10 Checkers, tự động tính toán `lower_bound` và `upper_bound` dựa trên lịch sử `previous_predictions`.
*   **Logit Bias Masking:** Hàm `get_logit_bias_for_bounds` sử dụng `tiktoken` để mã hóa các chuỗi số (từ `0.00` đến `1.00`) và gắn điểm phạt `-100` cho tất cả các token nằm ngoài khoảng an toàn trước khi chuyển kết quả lập luận cho parser.
*   **Softmax Fallback:** Trong trường hợp logit bias bị lỗi, hàm `call_with_tcd_sync` / `call_with_tcd_async` sẽ sử dụng hàm `np.clip` làm chốt chặn cuối cùng để cắt (clip) xác suất đầu ra về khoảng hợp lệ.

---

## 4. Tổng Hợp Các Model AI Đang Sử Dụng Qua API

Mọi hoạt động suy luận logic, đối kháng và phân tích kết quả dự báo trong dự án đều gọi qua Cloud API (Proxy được cấu hình tại [.env](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/.env)):

1.  **OpenAI Models (API: `https://llm.wokushop.com/v1`):**
    *   `gpt-4o-2024-05-13` và `gpt-4o-2024-08-06` (Dùng cho dự báo cơ bản và khởi tạo logic tuples).
    *   `gpt-4o-mini` (Model thu nhỏ của GPT-4o).
    *   `gpt-5.4-mini` (Model mặc định của hệ thống).
    *   `o1-mini` và `o1-preview` (Dòng mô hình suy luận sâu phục vụ chạy CoT).
    *   `o4-mini-all` (Mô hình suy luận thế hệ mới thuộc dòng o4).
2.  **MiniMax Models (API: `https://api.xah.io/v1`):**
    *   `tduckcontact/MiniMax-M3` (Mô hình ngôn ngữ lớn tiên tiến của MiniMax).
3.  **Anthropic Models (API: `https://llm.wokushop.com/v1`):**
    *   `claude-3.5-sonnet` (Chạy thực nghiệm đánh giá độ nhất quán logic và độ chính xác).
4.  **Meta Llama Models (Gọi qua OpenRouter/TogetherAI):**
    *   `llama-3.1-8B`, `llama-3.1-70B`, `llama-3.1-405B` (Đánh giá hiệu năng dự báo tăng dần theo tham số).
5.  **Perplexity Models (API: `https://api.perplexity.ai`):**
    *   `llama-3.1-sonar-large-128k-online` và `llama-3.1-sonar-huge-128k-online` (Dòng mô hình cập nhật tin tức trực tiếp phục vụ giải quyết kết quả câu hỏi nguồn).

---

## 5. Quy Trình Chạy Dự Án (Tóm Tắt Công Việc Cần Thực Hiện)

Để đạt được kết quả phân tích cuối cùng, bạn thực hiện chạy dự án theo quy trình sau:

### Bước 1: Setup Môi trường & API Key
1.  Kích hoạt môi trường ảo Python và cài đặt [requirements.txt](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/requirements.txt).
2.  Điền đầy đủ thông tin API key vào [.env](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/.env).

### Bước 2: Kiểm Tra & Gán Nhãn Dữ Liệu (Tùy chọn)
Chạy ứng dụng Streamlit [feedback_form.py](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/data_labeling/feedback_form.py) để tinh lọc các câu hỏi dự báo thô ở đầu vào.

### Bước 3: Chạy Thực Nghiệm Đánh Giá & So Sánh Hiệu Năng

Để đánh giá hiệu quả của phương pháp **HybridACD** so với phương pháp dự báo cơ bản ban đầu, chúng ta thực hiện chạy thực nghiệm đánh giá cho cả hai phương pháp (trước và sau cải tiến) trên hai khía cạnh: **Độ Nhất Quán (Consistency)** và **Độ Chính Xác (Ground Truth)**.

cd "d:\UIT Document\UIT subjects\DS391 - LLM\Project\consistency-forecasting"
.venv\Scripts\Activate.ps1


#### A. Đánh Giá Độ Nhất Quán (Consistency Evaluation)
Đo lường tần suất và mức độ vi phạm 10 quy tắc logic (arbitrage/frequentist metrics) trên tập câu hỏi tuples.

1.  **Trước cải tiến (Phương pháp gốc - BasicForecaster):**
    Chạy tập lệnh [src/evaluation.py](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/src/evaluation.py) sử dụng `BasicForecaster` với model mặc định đã nâng cấp:
    ```bash
    .\.venv\Scripts\python src/evaluation.py --tuple_dir src/data/tuples/scraped -f BasicForecaster -o model=mistral-medium-3.5-128b --num_lines 20 --run --async -k all --output_dir src/data/forecasts/Basic_consistency_mistral-medium-3.5-128b
    ```

2.  **Sau cải tiến (Phương pháp HybridACD - HybridACDForecaster):**
    Chạy tập lệnh [src/evaluation.py](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/src/evaluation.py) sử dụng `HybridACDForecaster`:
    ```bash
    .\.venv\Scripts\python src/evaluation.py --tuple_dir src/data/tuples/scraped -f HybridACDForecaster -o model=mistral-medium-3.5-128b --num_lines 20 --run --async -k all --output_dir src/data/forecasts/HybridACD_consistency_mistral-medium-3.5-128b
    ```

#### B. Đánh Giá Độ Chính Xác (Ground Truth Run)
Đối chiếu xác suất dự báo của mô hình với kết quả thực tế (ground truth) để tính toán Brier Score và vẽ biểu đồ hiệu chuẩn (calibration curve).

> [!NOTE]
> **Lưu ý 1: Tập dữ liệu Ground Truth quá nhỏ (chỉ có 3 câu hỏi)**
> * **Nguyên nhân cốt lõi**: Tùy chọn `--num_lines` (hoặc `-n`) mặc định bằng `3` trong [common_options.py](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/src/evaluation_utils/common_options.py#L76-L80) tự động cắt ngắn tập dữ liệu nếu không truyền giá trị `-1` một cách tường minh.
> * **Khắc phục (TIP)**: Truyền thêm cờ `--num_lines -1` (hoặc `-n -1`) vào lệnh chạy để hệ thống xử lý toàn bộ các câu hỏi (tập dữ liệu thực tế tại `src/data/fq/real/` có từ 182 đến 242 câu hỏi, trong đó có 134 câu hỏi đã được resolved).

> [!IMPORTANT]
> **Lưu ý 2: HybridACDForecaster thiếu file ground_truth_summary.json**
> * **Nguyên nhân cốt lõi**: Sự phân tách độc lập giữa luồng Nhất quán ([src/evaluation.py](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/src/evaluation.py) tạo ra `stats_summary.json`) và luồng Độ chính xác thực tế ([src/ground_truth_run.py](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/src/ground_truth_run.py) tạo ra `ground_truth_summary.json`). Lượt chạy trước chỉ mới chạy HybridACD trên luồng Nhất quán.
> * **Giải pháp**: Cần chạy bổ sung luồng Ground Truth cho `HybridACDForecaster` bằng cách chỉ định module cụ thể của HybridACD qua cờ `-p` (`-p src/forecasters/hybrid_acd_forecaster.py::HybridACDForecaster`).

1.  **Trước cải tiến (Phương pháp gốc - BasicForecaster):**
    Chạy tập lệnh [src/ground_truth_run.py](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/src/ground_truth_run.py) sử dụng `BasicForecaster`:
    ```bash
    .\.venv\Scripts\python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --forecaster_class BasicForecaster --forecaster_options model=mistral-medium-3.5-128b --num_lines -1 --run --async --output_dir src/data/forecasts/Basic_groundtruth_mistral-medium-3.5-128b
    ```

2.  **Sau cải tiến (Phương pháp HybridACD - HybridACDForecaster):**
    Chạy tập lệnh [src/ground_truth_run.py](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/src/ground_truth_run.py) sử dụng `HybridACDForecaster`:
    ```bash
    .\.venv\Scripts\python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -p src/forecasters/hybrid_acd_forecaster.py::HybridACDForecaster -o model=mistral-medium-3.5-128b --num_lines -1 --run --async --output_dir src/data/forecasts/HybridACD_groundtruth_mistral-medium-3.5-128b
    ```
`
### Bước 4: Tổng Hợp Số Liệu & Báo Cáo

1.  Mở Jupyter notebook [results_analysis.ipynb](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/results_analysis.ipynb) để tính tương quan Pearson và vẽ bản đồ nhiệt phân bổ độ nhất quán.
2.  Chạy script [src/plot_consistency_vs_brier.py](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/src/plot_consistency_vs_brier.py) để xuất đồ thị so sánh độ nhất quán logic và độ chính xác dự báo (Brier Score) của các mô hình:
    ```bash
    .\.venv\Scripts\python src/plot_consistency_vs_brier.py
    ```
