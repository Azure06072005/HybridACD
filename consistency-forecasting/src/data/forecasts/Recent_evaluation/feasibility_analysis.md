# Báo cáo Phân tích Tính khả thi & Giải pháp Khắc phục

Báo cáo này tập trung phân tích hai nhược điểm chí mạng được chỉ ra trong báo cáo so sánh trước đó giữa **BasicForecaster** và **HybridACDForecaster**:
1. **Tập dữ liệu Ground Truth quá nhỏ (chỉ có 3 câu hỏi)** nên chỉ số Brier Score chưa mang tính đại diện thống kê cao.
2. **HybridACDForecaster không có file `ground_truth_summary.json`** trong thư mục kết quả.

Dưới đây là phân tích chi tiết về nguyên nhân cốt lõi, khả năng thực thi và các bước khắc phục cụ thể cho từng vấn đề.

---

## 1. Phân tích Nhược điểm 1: Tập dữ liệu Ground Truth quá nhỏ (3 câu hỏi)

### Nguyên nhân cốt lõi
Trong mã nguồn CLI của hệ thống (file [common_options.py](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/src/evaluation_utils/common_options.py#L76-L80)), tùy chọn `--num_lines` (hoặc viết tắt là `-n`) dùng để giới hạn số lượng câu hỏi xử lý được đặt giá trị mặc định là `3`:
```python
click.option(
    "-n",
    "--num_lines",
    default=3,
    help="Number of lines to process in each of the files",
)
```
Trong script [src/ground_truth_run.py](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/src/ground_truth_run.py#L107-L109), nếu tham số `num_lines` không được đặt thành `-1` một cách tường minh, hệ thống sẽ giữ nguyên giá trị mặc định (`3`), dẫn đến việc tập dữ liệu Ground Truth bị cắt ngắn (prune) chỉ còn đúng 3 dòng đầu tiên trong quá trình chạy thực nghiệm.

### Khả năng khắc phục
> [!TIP]
> **Hoàn toàn khắc phục được 100%** và cực kỳ đơn giản mà không cần chỉnh sửa bất kỳ dòng mã nguồn nào của hệ thống.

### Giải pháp chi tiết
Tập dữ liệu Ground Truth thực tế trong thư mục `src/data/fq/real/` có kích thước khá lớn, chẳng hạn như:
- `20240501_20240815.jsonl` có **242 câu hỏi** (trong đó có 134 câu hỏi đã được giải quyết - resolved).
- `metaculus_cleaned_formatted_20240501_20240815.jsonl` có **182 câu hỏi**.

Để chạy đánh giá độ chính xác (Brier Score) trên toàn bộ tập dữ liệu này, ta chỉ cần chạy lệnh thực thi và truyền thêm tham số `--num_lines -1` (hoặc `-n -1`). Khi nhận giá trị `-1`, mã nguồn tại `src/ground_truth_run.py` sẽ gán `num_lines = None` và xử lý toàn bộ các câu hỏi có nhãn phân giải thực tế.

**Lệnh chạy khắc phục mẫu (cho BasicForecaster):**
```bash
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --forecaster_class BasicForecaster --forecaster_options model=gpt-5.4-mini --num_lines -1 --run --async --output_dir src/data/forecasts/Basic_groundtruth_run_full
```

---

## 2. Phân tích Nhược điểm 2: HybridACDForecaster thiếu file `ground_truth_summary.json`

### Nguyên nhân cốt lõi
Hệ thống đánh giá được chia làm hai luồng chạy hoàn toàn độc lập:
1. **Luồng Đánh giá Nhất quán logic (Consistency Evaluation)** thông qua script [src/evaluation.py](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/src/evaluation.py). Luồng này chạy trên tập câu hỏi Tuples và sinh ra file kết quả nhất quán `stats_summary.json`.
2. **Luồng Đánh giá Độ chính xác thực tế (Ground Truth Run)** thông qua script [src/ground_truth_run.py](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/src/ground_truth_run.py). Luồng này chạy trên tập câu hỏi có nhãn thực tế và sinh ra file đánh giá dự báo `ground_truth_summary.json`.

Ở lượt chạy trước, mô hình `HybridACDForecaster` mới chỉ được chạy qua luồng 1 (`src/evaluation.py`) để đánh giá các chỉ số vi phạm logic mà chưa được chạy qua luồng 2 (`src/ground_truth_run.py`). Do đó, trong thư mục kết quả của HybridACD chỉ có file `stats_summary.json` và hoàn toàn thiếu file `ground_truth_summary.json`.

### Khả năng khắc phục
> [!IMPORTANT]
> **Hoàn toàn khắc phục được 100%** bằng cách thực thi thêm bước chạy Ground Truth cho HybridACDForecaster theo đúng quy trình hệ thống.

### Giải pháp chi tiết
Để sinh file `ground_truth_summary.json` cho HybridACD, ta cần chạy script `src/ground_truth_run.py` và chỉ định class là `HybridACDForecaster` (sử dụng đường dẫn module qua cờ `-p` hoặc tên lớp qua `-f` nếu đã được định nghĩa trong `common_options.py`).

**Lệnh chạy khắc phục cụ thể:**
```bash
python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -p src/forecasters/hybrid_acd_forecaster.py::HybridACDForecaster -o model=gpt-5.4-mini --num_lines -1 --run --async --output_dir src/data/forecasts/HybridACD_groundtruth_run_full
```

*Lưu ý:*
- Cờ `-p src/forecasters/hybrid_acd_forecaster.py::HybridACDForecaster` giúp chỉ định đúng module chứa lớp tối ưu hóa HybridACD.
- Cờ `--num_lines -1` đảm bảo chạy trên toàn bộ tập dữ liệu thực tế thay vì bị giới hạn ở 3 câu hỏi như lượt chạy trước.
- Khi tiến trình chạy này hoàn tất, thư mục kết quả `src/data/forecasts/HybridACD_groundtruth_run_full` sẽ chứa đầy đủ cả file kết quả `ground_truth_results.jsonl`, file biểu đồ hiệu chuẩn `calibration_plot_linear.png`, và file tổng hợp `ground_truth_summary.json`.

---

## 3. Nhận định Tổng quan và Khuyến nghị Quy trình Chạy lại

Việc thực hiện hai giải pháp sửa lỗi trên không chỉ giúp khắc phục các hạn chế kỹ thuật của lượt chạy cũ, mà còn mang lại những giá trị phân tích quan trọng cho dự án:

1. **Ý nghĩa Khoa học**: Khi chạy đánh giá Ground Truth trên quy mô dữ liệu lớn (ví dụ: ~134 câu hỏi đã được giải quyết), chỉ số Brier Score sẽ phản ánh chính xác thực chất năng lực dự báo của mô hình.
2. **Kiểm tra Luật Goodhart**: Việc có đầy đủ kết quả Ground Truth của cả `BasicForecaster` và `HybridACDForecaster` trên cùng một tập dữ liệu lớn sẽ giúp chúng ta so sánh trực tiếp xem: **Cơ chế tối ưu hóa độ nhất quán logic (HybridACD) có vô tình làm suy giảm độ chính xác dự đoán thực tế (Brier Score) hay không?** 
   - Nếu HybridACD vừa giảm vi phạm logic (-97.8%) vừa giữ nguyên hoặc cải thiện Brier Score, điều đó chứng minh tính hiệu quả tuyệt đối của phương pháp.
   - Nếu Brier Score bị giảm sút nghiêm trọng, chúng ta sẽ cần điều chỉnh lại trọng số của logit bias.

### Khuyến nghị các bước chạy thực nghiệm hoàn chỉnh:
- **Bước 1**: Chạy đánh giá Ground Truth đầy đủ cho BasicForecaster:
  ```bash
  python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl --forecaster_class BasicForecaster --forecaster_options model=gpt-5.4-mini --num_lines -1 --run --async --output_dir src/data/forecasts/Basic_groundtruth_run_full
  ```
- **Bước 2**: Chạy đánh giá Ground Truth đầy đủ cho HybridACDForecaster:
  ```bash
  python src/ground_truth_run.py --input_file src/data/fq/real/20240501_20240815.jsonl -p src/forecasters/hybrid_acd_forecaster.py::HybridACDForecaster -o model=gpt-5.4-mini --num_lines -1 --run --async --output_dir src/data/forecasts/HybridACD_groundtruth_run_full
  ```
- **Bước 3**: Cập nhật lại các biểu đồ phân tích và so sánh hiệu năng trực quan trong Jupyter Notebook [compare_basic_vs_hybrid.ipynb](file:///d:/UIT Document/UIT subjects/DS391 - LLM/Project/consistency-forecasting/compare_basic_vs_hybrid.ipynb) dựa trên các thư mục kết quả mới.
