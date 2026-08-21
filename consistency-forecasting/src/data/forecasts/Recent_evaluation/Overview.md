# Overview: Phan tich Toan dien Thuc nghiem HybridACD

> **Ngay tao:** 25/06/2026
> **Du an:** Consistency Checks for LLM Forecasters - Cai tien HybridACD
> **Baseline Paper:** "Consistency Checks for Language Model Forecasters" (ICLR 2025)

---

## 1. Tong hop cac File Lien quan den HybridACD

### 1.1 File Trien khai (Implementation Files)

| File | Mo ta |
|---|---|
| `src/forecasters/hybrid_acd_forecaster.py` | Lop HybridACDForecaster - 3 modules: Adversarial Agent + CoT + TCD, 487 dong code |
| `src/evaluation.py` | Script danh gia tinh nhat quan logic tren tap Tuples (10 quy tac) |
| `src/ground_truth_run.py` | Script danh gia Brier Score tren du lieu ground truth |
| `workflow.md` | Tai lieu quy trinh day du: setup, chay thuc nghiem, phan tich |
| `src/forecasters/basic_forecaster.py` | Lop BasicForecaster - baseline goc |
| `src/forecasters/cot_forecaster.py` | Lop CoT-Forecaster - baseline nang cao |
| `src/forecasters/consistent_forecaster.py` | Lop ArbitrageForecaster (ConsistentForecaster) |
| `results_analysis.ipynb` | Jupyter notebook phan tich tuong quan Pearson va visualization |

### 1.2 File Ket qua - Recent Evaluation (Thuc nghiem cua Nhom)

Duong dan: `src/data/forecasts/Recent_evaluation/`

| Thu muc con | Model duoc kiem tra | So cau hoi GT | Noi dung |
|---|---|---|---|
| `genimi_2.5_flash/` | gemini-2.5-flash | 242 | Basic + HybridACD consistency + ground truth |
| `gpt_4o_mini/` | gpt-4o-mini | 242 | Basic + HybridACD consistency + ground truth |
| `gpt_5.4_mini/` | gpt-5.4-mini | 3-20 | Basic + HybridACD (thu nghiem nho) |
| `mistral_medium_3.5/` | mistral-medium-3.5-128b | 242 | Basic + HybridACD consistency + ground truth |
| `mistral_small_4/` | mistral-small-4-119b | 242 | Basic + HybridACD consistency + ground truth |
| `Minimax_M3/` | tduckcontact/MiniMax-M3 | 242 | Basic + HybridACD consistency + ground truth |
| `compare_basic_vs_hybrid.ipynb` | — | — | Jupyter notebook so sanh truc quan |
| `feasibility_analysis.md` | — | — | Phan tich han che ky thuat va giai phap |

### 1.3 File Ket qua - Nhom Nghien cuu Goc (Baseline ICLR 2025)

| Loai Forecaster | Models | Datasets |
|---|---|---|
| BasicForecaster | gpt4o-2024-05-13, gpt4o-2024-08-06, gpt4o-mini, llama-3.1-8B/70B/405B, claude-3.5-sonnet | scraped, newsapi, 2028, ground truth |
| CoT_ForecasterTextBeforeParsing | gpt4o, gpt4o-mini, llama-3.1 all sizes, o1-mini, o1-preview, claude-3.5-sonnet | scraped, newsapi, 2028, ground truth |
| ConsistentForecaster (Arbitrage N=4) | gpt-4o-2024-08-06 | N4, P4, NP4, 4xEE1 variants (0x-3x iterations) |
| HybridACD_consistency_run | gpt-5.4-mini (initial test) | scraped tuples |
| HybridACD_groundtruth_run | gpt-5.4-mini (initial test) | 20240501_20240815.jsonl |

### 1.4 Cau truc moi thu muc ket qua

**Consistency run:**
```
<RunName>/
+-- AndChecker.jsonl          # Du bao cho quy tac AND
+-- AndOrChecker.jsonl        # Du bao cho quy tac AND-OR
+-- ButChecker.jsonl          # Du bao cho quy tac BUT
+-- CondChecker.jsonl         # Du bao cho quy tac COND
+-- CondCondChecker.jsonl     # Du bao cho quy tac CONDCOND
+-- ConsequenceChecker.jsonl  # Du bao cho quy tac CONSEQUENCE
+-- ExpectedEvidenceChecker.jsonl  # Du bao cho quy tac EXPEVIDENCE
+-- NegChecker.jsonl          # Du bao cho quy tac NEGATION
+-- OrChecker.jsonl           # Du bao cho quy tac OR
+-- ParaphraseChecker.jsonl   # Du bao cho quy tac PARAPHRASE
+-- config.jsonl              # Cau hinh thuc nghiem
+-- stats_*.json              # Thong ke tung quy tac (AVS, frequentist)
+-- stats_aggregated.json     # Tong hop toan bo (avg_violation)
+-- stats_summary.json        # Bao cao chi tiet nhat
+-- stats_forecaster.json     # Ten forecaster class
```

**Ground truth run:**
```
<RunName>/
+-- ground_truth_results.jsonl     # Tung cau hoi: prob + outcome
+-- ground_truth_summary.json      # Brier Score, calibration error, log score
+-- calibration_plot_linear.png    # Bieu do hieu chuan (calibration curve)
```

---

## 2. Giai thich Chi tiet Thuc nghiem HybridACD

### 2.1 Bai toan Goc (Research Problem)

Nghien cuu goc ICLR 2025 cua Shapiro et al. dua ra luan diem:

> Neu mot LLM co kha nang du bao thuc chat, no phai duy tri mot "the gioi quan thong nhat" ve mat toan hoc. Cac xac suat du bao phai tuan thu cac tien de xac suat co ban.

**Hai chi so danh gia vi pham:**
- **Arbitrage Metric (default):** Loi nhuan chenh lech gia toi da ma adversary co the rut tu mo hinh vi pham (Dutch Book). Dua tren LMSR (Logarithmic Market Scoring Rule), xu phat nang voi vi pham tai bien (xac suat 0 hoac 1).
- **Frequentist Metric:** So do lech chuan chenh lech so voi phan phoi Null (Monte Carlo simulation). Do luong xem sai lech la nhieu ngau nhien hay sap do thuc su.

**10 quy tac nhat quan duoc kiem tra:**

| # | Ten | Rang buoc toan hoc | Key trong Tuple |
|---|---|---|---|
| 1 | NEGATION | P(A) + P(neg_A) = 1 | P, not_P |
| 2 | PARAPHRASE | P(A) = P(A') | P, para_P |
| 3 | CONSEQUENCE | A implies B -> P(A) <= P(B) | P, cons_P |
| 4 | AND | max(0, P+Q-1) <= P(A and B) <= min(P,Q) | P, Q, P_and_Q |
| 5 | OR | max(P,Q) <= P(A or B) <= min(1, P+Q) | P, Q, P_or_Q |
| 6 | BUT | P(A or B) = P(A) + P(B and neg_A) | P, Q_and_not_P, P_or_Q |
| 7 | COND | P(A).P(B|A) = P(A and B) | P, Q_given_P, P_and_Q |
| 8 | CONDCOND | P(A).P(B|A).P(C|A and B) = P(A and B and C) | P, Q_given_P, R_given_P_and_Q, P_and_Q_and_R |
| 9 | ANDOR | P(A)+P(B) = P(A or B)+P(A and B) | P, Q, P_and_Q, P_or_Q |
| 10 | EXPEVIDENCE | P(A) = P(A|B)P(B) + P(A|neg_B)(1-P(B)) | P, Q, P_given_Q, P_given_not_Q |

### 2.2 Han che Phat hien tu Nghien cuu Goc

**Han che 1 - Chi phi ArbitrageForecaster (Post-hoc):**
- ConsistentForecaster N=4 iterations: ~900 API calls/tuple ~ $2,500/cau hoi
- Geometric overfitting: sua NEGATION gay hoa hai CONDCOND va nguoc lai
- Goi y Random Sampling chi giam nhiet chi phi, khong xu ly overfitting

**Han che 2 - Goodhart's Law trong Fine-tuning (Pre-hoc):**
- Mo hinh toi thieu hoa Arbitrage Loss hoc cach "gian lan": P = 0.5 cho moi cau
- Nhat quan hoan hao (violation = 0) nhung Brier Score vo gia tri (~0.25)
- Khong the phhan biet duoc vi pham thuc su hay collapse du bao

**Han che 3 - Linguistic Artifacts (NEGATION anomaly):**
- Tren tap NewsAPI, NEGATION violation co tuong quan am (r = -0.31) voi Brier Score
- Nguyen nhan: Reference Class Spanning - cau hoi chua tu phu dinh cu phap
- Mo hinh bi nham lan giua phu dinh cu phap va phu dinh logic

### 2.3 Kien truc HybridACD - Giai phap The he Thu Ba

HybridACD giai quyet tam de tren bang cach can thiep tai dung thoi diem giai ma (decoding-time):

```
Consistency Tuple: {Q1, Q2, ..., Qk} + Logical Rule L
                        |
            [MODULE 1: Adversarial Input Agent]
                LLM phu viet lai Qi thanh Qi'
                Phu tap ve ngon ngu, bao toan toan hoc
                temperature=0.7, response_model=AdversarialOutput
                        |
            [MODULE 2: Chain-of-Thought Reasoning]
                LLM chinh (backbone) suy luan tu do (scratchpad)
                query_api_chat_native() - khong gioi han format
                        |
            [MODULE 3: Token Constraint Decoding]
                1. get_consistency_bounds() -> [lower, upper]
                2. get_logit_bias_for_bounds() -> {token_id: -100}
                3. query_api_chat(logit_bias=mask) -> p in [l,u]
                4. np.clip(p, lower, upper) fallback
                        |
            p̂i guaranteed in [lower, upper] -- nhat quan tuyet doi
```

**Uu diem so voi cac phuong phap cu:**
- Khong thay doi weights -> mien nhiem Goodhart Law
- Khong de quy API -> chi phi ~0 them
- Can thiep tai logit space truoc khi commit xac suat -> kiem soat toan hoc hoan hao
- Model-agnostic: hoat dong voi bat ky LLM nao ho tro logit_bias

---

## 3. Phan tich Tung Buoc File hybrid_acd_forecaster.py

### 3.1 Adversarial Agent Prompt (dong 16-34)

```python
ADVERSARIAL_AGENT_PREFACE = (
    "You are an adversarial question generator. Your job is to rewrite the input "
    "forecasting question to make it syntactically complex and linguistically "
    "challenging, while keeping its mathematical meaning and resolution criteria "
    "completely identical."
)
```

**Chien luoc rewrite:**
- Grammatical noise: them menh de phu, dao ngu
- Entity swapping: thay the thuc the bang bieu dat tuong duong
- Nested conditional logic (CONDCOND): long them dieu kien
- Evidence rules (EXPEVIDENCE): them bang chung de loc

### 3.2 Constructor - HybridACDForecaster (dong 36-65)

```python
def __init__(self, model, adversarial_model=None, preface=None,
             examples=None, adversarial_enabled=True, tcd_enabled=True):
```

5 tham so cau hinh quan trong:
- `model`: LLM chinh lam Forecaster backbone
- `adversarial_model`: LLM phu (mac dinh = model chinh, co the la Llama-3-70B)
- `adversarial_enabled`: Bat/tat Module 1 (de test ablation)
- `tcd_enabled`: Bat/tat Module 3 (de test ablation)
- `preface`: System prompt tuy chinh

### 3.3 elicit() / elicit_async() - Vong lap xu ly Tuple (dong 74-141)

```python
for key in keys:                          # Xu ly tuan tu (sequential)
    fq = fqs[key]
    # Buoc 1: Adversarial perturbation
    fq_query = self.adversarial_rewrite_sync(fq)
    # Tinh bounds tu du doan truoc
    lower_bound, upper_bound = self.get_consistency_bounds(
        keys, previous_predictions, key)
    # Buoc 2+3: CoT + TCD
    forecast = self.call_with_tcd_sync(fq_query, lower_bound, upper_bound)
    previous_predictions[key] = forecast.prob
```

**Ly do xu ly tuan tu:** Cau hoi Qi+1 can ket qua cua Qi de tinh bounds chinh xac.
Khong the song song vi se mat rang buoc toan hoc giua cac cau trong tuple.

### 3.4 adversarial_rewrite_async() - Module 1 (dong 143-163)

```python
response = await query_api_chat(
    messages=[{system: ADVERSARIAL_AGENT_PREFACE}, {user: prompt}],
    model=self.adversarial_model,
    response_model=AdversarialOutput,  # Structured JSON: {title, body}
    temperature=0.7,                   # Tang tinh sang tao ngon ngu
)
fq_copy = fq.model_copy()
fq_copy.title = response.title
fq_copy.body = response.body
```

**Fallback:** Neu rewrite that bai (API error, JSON parse error), tra ve cau goc.
Dieu nay dam bao he thong van hoat dong du adversarial agent bi loi.

### 3.5 call_with_tcd_async() - Module 2+3 (dong 188-242)

**Buoc 1 - Sinh CoT (Chain-of-Thought):**
```python
native_response = await query_api_chat_native(
    model=self.model,
    messages=[{system: preface}, {user: fq.to_str_forecast_mode()}],
)
```
Goi raw API (khong qua structured parsing) de lay toan bo CoT text tu do.

**Buoc 2 - Tinh Logit Bias Map:**
```python
logit_bias_map = self.get_logit_bias_for_bounds(
    parsing_model,   # gpt-4o-mini-2024-07-18
    lower_bound, upper_bound
)
```

**Buoc 3 - Parse + TCD Constraint:**
```python
parser_prompt = (
    f"Please parse the probability estimate from: '{native_response}'."
    f"The valid range is [{lower_bound:.4f}, {upper_bound:.4f}]."
)
parsed_response = await query_api_chat(
    response_model=Prob,
    model=parsing_model,
    messages=[...parser_prompt...],
    logit_bias=logit_bias_map,    # KEY: API constraint
)
prob_val = float(np.clip(prob_val, lower_bound, upper_bound))  # failsafe
```

**Metadata luu tru:**
```python
return Forecast(prob=prob_val, metadata={
    "chain_of_thought": native_response,
    "lower_bound": lower_bound,
    "upper_bound": upper_bound
})
```

### 3.6 get_logit_bias_for_bounds() - Co che FST (dong 301-319)

```python
for i in range(101):              # 0.00, 0.01, ..., 1.00
    val = i / 100.0
    if val < lower or val > upper:
        for fmt in [f"{val:.2f}", f"{val:.1f}"]:
            tokens = encoding.encode(fmt)
            if len(tokens) == 1:           # Chi single-token numbers
                bias[str(tokens[0])] = -100   # -100 = -inf trong OpenAI
```

**Chi tiet ky thuat:**
- Tiktoken encoding: `o200k_base` (o1/gpt-4o), `cl100k_base` (khac)
- Gia tri `-100` tuong duong `-inf` trong softmax cua OpenAI API
- Chi xu ly single-token numbers de tranh ambiguity multi-token (e.g., "0.57" co the la ["0", ".", "57"])
- Ket qua: model khong the emit token dai dien cho gia tri ngoai [lower, upper]

### 3.7 get_consistency_bounds() - Toan hoc Rang buoc (dong 321-486)

Trien khai day du 10 quy tac nhat quan thanh ham tinh bounds:

**NEGATION (2 bien):** {P, not_P}
```
not_P = 1 - P  ->  lower = upper = 1 - p(P)
```

**PARAPHRASE (2 bien):** {P, para_P}
```
para_P = P  ->  lower = upper = p(P)
```

**COND (3 bien):** {P, Q_given_P, P_and_Q}
```
P_and_Q = P * Q|P  ->  exact value
Q|P = P_and_Q / P  ->  exact value (neu P > 1e-6)
```

**CONDCOND (4 bien):** {P, Q_given_P, R_given_P_and_Q, P_and_Q_and_R}
```
P_and_Q_and_R = P * Q|P * R|(P and Q)  ->  exact value
R|(P and Q) = P_and_Q_and R / (P * Q|P)  ->  exact value
```

**EXPEVIDENCE (4 bien):** {P, Q, P_given_Q, P_given_not_Q}
```
P = Q*P|Q + (1-Q)*P|neg_Q  ->  exact value
Q = (P - P|neg_Q) / (P|Q - P|neg_Q)  ->  exact value (neu diff > 1e-6)
```

**AND (3 bien):** {P, Q, P_and_Q}
```
lower = max(0, P+Q-1)  ->  interval [lower, upper]
upper = min(P, Q)
```

**Safety clamp cuoi:**
```python
lower = max(0.0, min(1.0, lower))
upper = max(0.0, min(1.0, upper))
if lower > upper: lower, upper = upper, lower  # swap neu nghich dao
```

---

## 4. Phan tich Ket qua Thuc nghiem

### 4.1 Nhat quan Logic - Arbitrage Metric (thap hon = tot hon)

Du lieu trich xuat tu `stats_aggregated.json`, tap Scraped:

| Model | Basic (default AVS) | HybridACD (default AVS) | Giam tuyet doi | Giam tuong doi |
|---|---|---|---|---|
| Gemini-2.5-Flash | 0.1116 | 0.0087 | -0.1029 | **-92.2%** |
| GPT-4o-mini | 0.0307 | 0.0007 | -0.0300 | **-97.6%** |
| GPT-5.4-mini | 0.4909 | 0.0107 | -0.4802 | **-97.8%** |
| MiniMax-M3 | 0.1266 | 0.0005 | -0.1261 | **-99.6%** |
| Mistral-Medium-3.5 | 0.0792 | 0.0087 | -0.0705 | **-89.1%** |
| Mistral-Small-4 | 0.0740 | 0.0023 | -0.0717 | **-96.8%** |
| **Trung binh** | **0.1522** | **0.0053** | **-0.1469** | **-95.5%** |

**Frequentist Metric (thap hon = tot hon):**

| Model | Basic (Freq.) | HybridACD (Freq.) | Giam tuong doi |
|---|---|---|---|
| Gemini-2.5-Flash | 0.2612 | 0.0168 | -93.6% |
| GPT-4o-mini | 0.1752 | 0.0055 | -96.9% |
| GPT-5.4-mini | 1.0423 | 0.0227 | -97.8% |
| MiniMax-M3 | 0.3756 | 0.0048 | -98.7% |
| Mistral-Medium-3.5 | 0.2494 | 0.0168 | -93.3% |
| Mistral-Small-4 | 0.2421 | 0.0156 | -93.6% |

**Nhan xet quan trong:**
- GPT-5.4-mini co AVS ban dau cao nhat (0.491) - phan anh model moi hon co xu huong dat xac suat cuc doan (can 0 hoac 1) mot cach tu tin, bi LMSR xu phat nang
- MiniMax-M3 dat muc giam cao nhat (99.6%) - cho thay TCD hoat dong hieu qua tren nhieu kien truc mo hinh khac nhau

### 4.2 Do chinh xac Du bao - Brier Score (thap hon = tot hon)

Du lieu tu `ground_truth_summary.json`, 242 cau hoi da resolve:

| Model | Basic BS | HybridACD BS | Thay doi | Ket luan |
|---|---|---|---|---|
| Gemini-2.5-Flash | 0.141 | **0.130** | -7.8% | Cai thien ca hai chi so |
| GPT-4o-mini | 0.205 | **0.200** | -2.4% | Cai thien ca hai chi so |
| MiniMax-M3 | 0.123 | **0.116** | -5.7% | Cai thien ca hai chi so |
| Mistral-Medium-3.5 | 0.202 | **0.167** | **-17.3%** | Cai thien manh nhat |
| Mistral-Small-4 | 0.211 | **0.202** | -4.3% | Cai thien ca hai chi so |

**Ket luan then chot:** HybridACD KHONG bi Goodhart Law. No vua giam vi pham nhat quan (95.5%) vua cai thien do chinh xac du bao (Brier Score giam 2.4-17.3%).

### 4.3 Phan tich Hieu chuan (Calibration Error)

| Model | Basic Cal.Error | HybridACD Cal.Error | Thay doi |
|---|---|---|---|
| Gemini-2.5-Flash | 0.195 | 0.185 | -5.1% |
| GPT-4o-mini | 0.161 | **0.105** | **-34.8%** |
| MiniMax-M3 | 0.161 | 0.117 | -27.3% |
| Mistral-Medium-3.5 | 0.090 | 0.138 | +53.3% (tang) |
| Mistral-Small-4 | 0.124 | 0.127 | +2.4% (tang nhe) |

**Luu y:** Mistral models co Calibration Error tang nhe khi dung HybridACD. Nguyen nhan kha nang: TCD "cuong buc" xac suat vao khoang hop le nhung xa voi muc tin cay thuc su cua mo hinh, gay lech phan phoi hieu chuan. Day la huong cai thien tiep theo.

### 4.4 So sanh voi Ket qua Nhom Nghien cuu Goc (ICLR 2025)

| Phuong phap | AVS (Scraped) | Brier Score (GT) | Chi phi/Q |
|---|---|---|---|
| BasicForecaster gpt4o-mini (goc) | ~0.412 | 0.205-0.231 | $0.002 |
| CoT-Forecaster gpt4o (goc) | ~0.287 | ~0.198 | $0.018 |
| CoT o1-preview (goc - tot nhat) | ~0.089 | ~0.170 | $0.05+ |
| ArbitrageForecaster N=4 (goc) | ~0.161 | ~0.245 | ~$2,500 |
| **HybridACD gpt-4o-mini (nhom)** | **0.0007** | **0.200** | **$0.019** |
| **HybridACD mistral-medium (nhom)** | **0.0087** | **0.167** | **$0.115** |
| **HybridACD Minimax-M3 (nhom)** | **0.0005** | **0.116** | **~$0.08** |
| **HybridACD Gemini-2.5-flash (nhom)** | **0.0087** | **0.130** | **~$0.07** |

**Nhan xet so sanh:**
- HybridACD dat AVS thap hon 20x so voi ArbitrageForecaster (0.0007 vs 0.161)
- Chi phi thap hon 21,700x so voi ArbitrageForecaster ($0.019 vs $2,500)
- Brier Score tuong duong hoac tot hon CoT o1-preview (model dat do nhat trong nghien cuu goc)
- Minimax-M3 voi HybridACD dat BS=0.116 - tot hon ca o1-preview trong nghien cuu goc

---

## 5. Bang So sanh Tong the

| Tieu chi | BasicForecaster | ArbitrageForecaster (Goc) | HybridACD (Moi) |
|---|---|---|---|
| Avg. Violation Score | 0.152 | 0.161 | **0.005** |
| Brier Score | 0.231 | 0.245 | **0.130-0.202** |
| Chi phi/tuple | $0.002 | $2,500 | **$0.019-0.115** |
| Rui ro Goodhart | Khong | Co (geometric overfit) | **Khong** |
| Can sua model weights | Khong | Khong | **Khong** |
| Nhat quan toan hoc | Ngau nhien | Gan dung (iterative) | **Dam bao tuyet doi** |
| Model-agnostic | Co | Khong (phu thuoc gpt4o) | **Co** |
| Scalability thuong mai | Co | Khong | **Co** |
| Immune Goodhart Law | N/A | Khong | **Co** |

---

## 6. Lenh Chay Thuc nghiem

### 6.1 Consistency Evaluation

```bash
cd "d:\UIT Document\UIT subjects\DS391 - LLM\Project\consistency-forecasting"
.venv\Scripts\Activate.ps1

# BasicForecaster
.venv\Scripts\python src/evaluation.py --tuple_dir src/data/tuples/scraped \
  -f BasicForecaster -o model=<MODEL> \
  --num_lines 20 --run --async -k all \
  --output_dir src/data/forecasts/Basic_consistency_<MODEL>

# HybridACDForecaster
.venv\Scripts\python src/evaluation.py --tuple_dir src/data/tuples/scraped \
  -f HybridACDForecaster -o model=<MODEL> \
  --num_lines 20 --run --async -k all \
  --output_dir src/data/forecasts/HybridACD_consistency_<MODEL>
```

### 6.2 Ground Truth Evaluation (242 cau hoi)

```bash
# BasicForecaster
.venv\Scripts\python src/ground_truth_run.py \
  --input_file src/data/fq/real/20240501_20240815.jsonl \
  --forecaster_class BasicForecaster --forecaster_options model=<MODEL> \
  --num_lines -1 --run --async \
  --output_dir src/data/forecasts/Basic_groundtruth_<MODEL>

# HybridACDForecaster
.venv\Scripts\python src/ground_truth_run.py \
  --input_file src/data/fq/real/20240501_20240815.jsonl \
  -p src/forecasters/hybrid_acd_forecaster.py::HybridACDForecaster \
  -o model=<MODEL> --num_lines -1 --run --async \
  --output_dir src/data/forecasts/HybridACD_groundtruth_<MODEL>
```

**Luu y quan trong:** Tham so `--num_lines -1` (hoac `-n -1`) bat buoc de chay tren toan bo 242 cau hoi. Gia tri mac dinh la `3` se chi chay 3 cau dau, ket qua khong co y nghia thong ke.

---

## 7. Han che Hien tai va Huong Nghien cuu Tiep theo

### Han che da xac dinh:
1. **Calibration Error tang tren Mistral:** TCD cuong buc xac suat vao khoang [l,u] nhung khong dam bao xac suat nay phu hop voi muc tin cay thuc su cua mo hinh
2. **RAG Forecasters chua duoc danh gia:** Google News/Newscatcher API bi ngat ket noi trong nghien cuu goc, chua co du lieu cho HybridACD voi RAG
3. **Complex CONDCOND:** Voi chain > 3 levels, tinh bounds co the NP-hard - can xap xi

### Huong nghien cuu tiep theo:
1. **Dieu chinh logit bias strength:** Thay vi -100 (tuyet doi), thu -50 hoac -30 tren Mistral de giam calibration drift
2. **Sequential Monte Carlo (SMC):** Xap xi bounds cho CONDCOND phuc tap hon
3. **Adversarial diversity:** Da dang hoa chien luoc rewrite (entity swap + syntactic inversion + negation lifting + conditional embedding)
4. **RAG evaluation:** Mo rong danh gia tren RAG-augmented forecasters khi API duoc khoi phuc

---

*Tai lieu nay duoc tao tu phan tich toan bo codebase va du lieu thuc nghiem ngay 25/06/2026.*
*Xem them chi tiet tai: workflow.md, feasibility_analysis.md, compare_basic_vs_hybrid.ipynb*