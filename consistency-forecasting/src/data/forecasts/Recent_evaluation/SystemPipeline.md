# System Pipeline Analysis: Consistency Forecasting + HybridACD

> **Ngay phan tich:** 25/06/2026
> **Pham vi:** Toan bo codebase `consistency-forecasting/src/` va ket qua thuc nghiem
> **Muc tieu:** Phan tich pipeline hoat dong cua he thong va tat ca file anh huong den ket qua

---

## 1. Kien truc Tong the He thong (4 Layers)

```
consistency-forecasting/
+-- src/
|   +-- common/                      [LAYER 1] Nen tang
|   |   +-- datatypes.py             Dinh nghia tat ca data model (Pydantic)
|   |   +-- llm_utils.py             Giao tiep voi moi LLM API
|   |   +-- perscache.py             Persistent cache (Redis/Local)
|   |   +-- utils.py                 Tien ich chung
|   +-- forecasters/                 [LAYER 2] Logic du bao
|   |   +-- forecaster.py            Abstract base class Forecaster
|   |   +-- basic_forecaster.py      BasicForecaster (baseline)
|   |   +-- cot_forecaster.py        CoT-Forecaster
|   |   +-- consistent_forecaster.py ArbitrageForecaster (post-hoc)
|   |   +-- hybrid_acd_forecaster.py HybridACDForecaster [CAI TIEN MOI]
|   |   +-- create.py                Factory khoi tao bat ky forecaster
|   +-- static_checks/               [LAYER 3] 10 quy tac nhat quan
|   |   +-- Checker.py               Abstract Checker + 10 checker classes
|   |   +-- MiniInstantiator.py      Tao cau hoi bien the (Neg, And, Or...)
|   |   +-- checker_prompts.py       Prompt templates cho tung checker
|   +-- evaluation_utils/            [LAYER 4] Cong cu danh gia
|   |   +-- common_options.py        CLI options cho moi script
|   |   +-- proper_scoring.py        Brier Score, Log Score, Calibration
|   +-- evaluation.py                [ENTRY 1] Danh gia nhat quan logic
|   +-- ground_truth_run.py          [ENTRY 2] Danh gia Brier Score
|   +-- forecaster_metrics.py        Cau hinh pairs de ve bieu do
|   +-- plot_consistency_vs_brier.py Ve bieu do scatter Pearson
|   +-- data/
|       +-- fq/real/                 242 cau hoi ground truth
|       +-- tuples/scraped/          Consistency tuples (10 loai)
|       +-- forecasts/               Ket qua thuc nghiem
|           +-- Recent_evaluation/   Ket qua moi nhat cua nhom
+-- results_analysis.ipynb           [ENTRY 3] Phan tich tuong quan Pearson
```

---

## 2. Phan tich File Theo Muc Do Anh Huong

### TIER 1: File Cot loi (Anh huong TRUC TIEP)

#### src/forecasters/hybrid_acd_forecaster.py [QUAN TRONG NHAT]
- **Kich thuoc:** 23,340 bytes, 487 dong
- **Vai tro:** Toan bo logic HybridACD (Module 1 + 2 + 3)

| Thanh phan | Dong | Mo ta |
|---|---|---|
| ADVERSARIAL_AGENT_PREFACE | 16-20 | System prompt cho adversarial agent |
| ADVERSARIAL_AGENT_PROMPT | 22-34 | Template prompt tao cau hoi doi khang |
| HybridACDForecaster.__init__ | 36-61 | Khoi tao, 5 tham so cau hinh |
| elicit() / elicit_async() | 74-141 | Vong lap chinh, xu ly TUAN TU tung cau |
| adversarial_rewrite_async() | 143-163 | Module 1: LLM viet lai (temperature=0.7) |
| call_with_tcd_async() | 188-242 | Module 2+3: CoT sinh + TCD constraint |
| get_logit_bias_for_bounds() | 301-319 | Co che FST: tiktoken + logit_bias=-100 |
| get_consistency_bounds() | 321-486 | 10 quy tac toan hoc thanh bounds [l,u] |

- **Anh huong:** AVS giam 95.5%, Brier Score cai thien 2.4-17.3%

---

#### src/evaluation.py [ENTRY POINT CHINH]
- **Kich thuoc:** 23,564 bytes, 641 dong
- **Vai tro:** Chay danh gia tinh nhat quan logic, tong hop stats_aggregated.json

Luong xu ly:
```
CLI args -> parse -> make_forecaster -> loop 10 checkers:
    doc tuples -> forecaster.elicit(tuple) -> tinh violations
    -> ghi <checker>.jsonl + stats_*.json + stats_aggregated.json
```

Ham quan trong:
- `get_stats()` dong 39-109: Tinh avg_violation, num_violations, median
- Metrics: `["default", "frequentist", "default_scaled"]`

---

#### src/ground_truth_run.py [ENTRY POINT BRIER]
- **Kich thuoc:** 16,060 bytes, 426 dong
- **Vai tro:** Danh gia Brier Score tren 242 cau hoi co ground truth

**BUG QUAN TRONG:** `--num_lines` mac dinh la `3` (dong 79 common_options.py)
Phai truyen `--num_lines -1` de chay toan bo 242 cau!

Luong xu ly:
```
doc 20240501_20240815.jsonl -> loop tung cau:
    pre_call() xoa resolution -> forecaster.call_full(fq)
    -> tinh brier_score, log_score -> ghi ground_truth_results.jsonl
-> tong hop ground_truth_summary.json + ve calibration_plot_linear.png
```

---

#### src/static_checks/Checker.py [10 QUY TAC]
- **Kich thuoc:** 76,696 bytes, 2118 dong
- **Vai tro:** Dinh nghia 10 Checker classes + thuat toan LMSR Arbitrage

| Checker Class | Tuple Keys | Constraint |
|---|---|---|
| NegChecker | {P, not_P} | P + not_P = 1 |
| ParaphraseChecker | {P, para_P} | P = para_P |
| ConsequenceChecker | {P, cons_P} | P <= cons_P |
| AndChecker | {P, Q, P_and_Q} | Frechet bounds |
| OrChecker | {P, Q, P_or_Q} | Frechet bounds |
| ButChecker | {P, Q_and_not_P, P_or_Q} | Inclusion-exclusion |
| CondChecker | {P, Q_given_P, P_and_Q} | Chain rule |
| CondCondChecker | {P, Q_given_P, R_given_P_and_Q, P_and_Q_and_R} | Nested chain rule |
| AndOrChecker | {P, Q, P_and_Q, P_or_Q} | Additivity |
| ExpectedEvidenceChecker | {P, Q, P_given_Q, P_given_not_Q} | Total probability |

Thuat toan tinh Arbitrage Metric: `scipy.optimize.minimize` LMSR
Frequentist metric: Monte Carlo z-score (sigma=0.05, gamma=2.58, beta=1e-3)

---

#### src/common/datatypes.py [DATA MODEL]
- **Kich thuoc:** 10,221 bytes, 397 dong
- **Vai tro:** Dinh nghia tat ca data model Pydantic, xuyen suot he thong

| Class | Mo ta |
|---|---|
| ForecastingQuestion | id, title, body, resolution_date, resolution |
| Forecast | prob (float [0,1]), metadata |
| Prob | prob (float [0,1]) voi validator |
| Prob_cot | chain_of_thought (str) + prob |
| ForecastingQuestion_stripped | Chi title + body |

Co che: `@field_validator("prob")` dam bao 0.0 <= prob <= 1.0 tren moi class

---

### TIER 2: File Quan trong (Anh huong GIAN TIEP)

#### src/common/llm_utils.py [API GATEWAY]
- **Kich thuoc:** 59,539 bytes, 1753 dong
- **Vai tro:** Diem duy nhat giao tiep voi tat ca LLM API

| Ham | Vai tro |
|---|---|
| query_api_chat() | Async + structured output + **logit_bias** support |
| query_api_chat_native() | Async raw (khong structured) - lay CoT text |
| answer() / answer_sync() | High-level wrapper, with_parsing=True |
| parallelized_call() | Async gather voi semaphore (max 25) |

LLM duoc ho tro: OpenAI (gpt-4o, gpt-4o-mini, gpt-5.4-mini, o1),
Anthropic (claude-3.5-sonnet), Meta/Llama (via Together/Groq),
Mistral (medium-3.5-128b, small-4-119b), Gemini, MiniMax (via OpenRouter)

#### src/common/perscache.py [CACHE]
- **Kich thuoc:** 25,622 bytes, 796 dong
- **Vai tro:** Cache persistent tat ca LLM calls (tranh goi API trung lap)

Co che: Key = SHA-256(model + messages + response_model + temperature)
Backend: Redis (production) hoac LocalFileStorage (development)
Dang ky: `register_model_for_cache(ModelClass)` trong datatypes.py

**Anh huong:** Bat = reproducible; Tat (NO_CACHE=True) = moi lan khac nhau;
Adversarial rewrite voi temperature=0.7 + cache miss = ket qua thay doi

#### src/evaluation_utils/proper_scoring.py [METRICS]
- **Kich thuoc:** 12,048 bytes, 364 dong

| Ham | Mo ta |
|---|---|
| brier_score(prob, outcome) | (prob - outcome)^2 |
| log_score(prob, outcome) | -log2(prob + eps) |
| scale_brier_score(bs) | Linear: 0.25->0, 0.00->100 |
| decompose_brier_score() | Uncertainty + Reliability + Resolution |
| calculate_calibration() | ECE (Expected Calibration Error) |
| platt_scaling() | Hoi quy logistic hieu chuan xac suat |

#### src/evaluation_utils/common_options.py [CLI]
- **Kich thuoc:** 5,175 bytes, 152 dong
- **BUG:** `--num_lines` default=3 (dong 79) - nguyen nhan thieu du lieu

| Tham so | Mac dinh | Anh huong |
|---|---|---|
| -f/--forecaster_class | None | Loai forecaster |
| -p/--custom_path | None | Custom class via dynamic import |
| -o/--forecaster_options | [] | model=X va cac kwargs |
| -n/--num_lines | **3 (!)** | Phai dat -1 cho full dataset |
| --async | False | Bat async mode (nhanh hon 20x) |
| --output_dir | auto | Thu muc luu ket qua |

#### src/forecasters/create.py [FACTORY]
- **Kich thuoc:** 6,856 bytes, 183 dong
- **Vai tro:** make_forecaster() - tao forecaster tu class name + config
- `"HybridACDForecaster"` da duoc dang ky trong `PREDEFINED_FORECASTER_CLASSES`

#### src/static_checks/MiniInstantiator.py [TUPLE GENERATOR]
- **Kich thuoc:** 85,298 bytes
- **Vai tro:** Tao cau hoi bien the cho tung consistency rule

| Instantiator | Tac dung |
|---|---|
| Neg | Tao cau phu dinh tu cau goc |
| And | Tao "A AND B" tu hai cau |
| Or | Tao "A OR B" |
| Conditional | Tao "B GIVEN A" |
| Paraphrase | Viet lai giu nguyen nghia |
| Consequence | Tao cau he qua tu tien de |

#### src/static_checks/checker_prompts.py [PROMPTS]
- **Kich thuoc:** 33,985 bytes
- **Vai tro:** Toan bo prompt templates cho viec tao cau hoi bien the

---

### TIER 3: File Ho tro (Visualization va Data)

| File | Kich thuoc | Vai tro |
|---|---|---|
| src/forecaster_metrics.py | 26,655 bytes | Cau hinh ForecasterPair cho bieu do |
| src/plot_consistency_vs_brier.py | 27,526 bytes | Bieu do scatter AVS vs Brier (Pearson) |
| results_analysis.ipynb | 902,906 bytes | Notebook phan tich, clustering |
| src/data/fq/real/20240501_20240815.jsonl | - | 242 cau hoi ground truth (binary outcome) |

---

## 3. Pipeline HybridACD Chi tiet Tung Buoc

### Consistency Evaluation Pipeline

```
[STEP 1] Khoi tao
python src/evaluation.py
  --tuple_dir src/data/tuples/scraped
  -f HybridACDForecaster
  -o model=<MODEL>
  --num_lines 20       <- so tuples moi checker
  --run --async -k all
  --output_dir <output>

         |
         v [common_options.py: parse CLI]
         |
         v [create.py: make_forecaster("HybridACDForecaster", {"model": M})]
         |
         v HybridACDForecaster(model=M, adversarial_enabled=True, tcd_enabled=True)

[STEP 2] Loop 10 Checkers: NegChecker, AndChecker, OrChecker, ...
  -> Doc tuples tu src/data/tuples/scraped/<Checker>.jsonl
  -> Lay top num_lines tuples

[STEP 3] Voi moi consistency tuple {Q1(key1), Q2(key2), ..., Qk(keyk)}

  history = {}
  for key in keys:     <- SEQUENTIAL (phai tuan tu)
    Qi = fqs[key]

    # MODULE 1: Adversarial Input Agent
    Qi_prime = adversarial_rewrite_async(Qi)
      -> query_api_chat(model=M, response_model=AdversarialOutput, temp=0.7)
      -> Qi_prime = {title: "...", body: "..."}  <- phuc tap ngon ngu, giu toan hoc

    # BOUND COMPUTATION
    [l, u] = get_consistency_bounds(keys, history, key)
      -> phan tich keys_set de nhan dang rule
      -> Vi du: keys_set = {"P", "not_P"} -> NEGATION -> l = u = 1 - history["P"]
      -> Vi du: keys_set = {"P", "Q", "P_and_Q"} -> AND -> l=max(0,p+q-1), u=min(p,q)
      -> Safety clamp: l = max(0, min(1, l)), u = max(0, min(1, u))

    # MODULE 2: Chain-of-Thought Reasoning
    cot_text = query_api_chat_native(model=M, messages=Qi_prime)
      -> Raw API call, khong structured format
      -> LLM tu do suy luan, viet reasoning chain

    # MODULE 3: Token Constraint Decoding
    bias_map = get_logit_bias_for_bounds("gpt-4o-mini-2024-07-18", l, u)
      -> for val in {0.00, 0.01, ..., 1.00}:
           if val < l or val > u:
             token_id = tiktoken.encode(str(val))
             if single_token: bias_map[token_id] = -100  <- -inf trong softmax

    p_hat = query_api_chat(
      model="gpt-4o-mini-2024-07-18",
      messages=[cot_text_as_context, "extract probability"],
      logit_bias=bias_map  <- HARD BLOCK tokens ngoai [l, u]
    )
    p_hat = clip(p_hat, l, u)  <- failsafe

    history[key] = p_hat
    result[key] = Forecast(prob=p_hat, metadata={cot, l, u})

[STEP 4] Tinh Violation
  default_metric: LMSR Arbitrage = scipy.optimize.minimize(portfolio_profit)
  frequentist: Monte Carlo z-score (sigma=0.05, gamma=2.58, beta=1e-3)

[STEP 5] Luu Ket qua
<output>/
+-- NegChecker.jsonl          {line:{P:{question,forecast}, not_P:{...}}, violation_data}
+-- stats_NegChecker.json     {avg_violation, num_violations, ...}
+-- stats_aggregated.json     {"default": {"avg_violation": 0.0087}, ...}
+-- stats_summary.json        Full detail tat ca metrics + config
+-- stats_forecaster.json     {"forecaster": "HybridACDForecaster"}
+-- config.jsonl              Cau hinh thuc nghiem day du
```

### Ground Truth Pipeline (Brier Score)

```
[STEP 1] Khoi tao
python src/ground_truth_run.py
  --input_file src/data/fq/real/20240501_20240815.jsonl   <- 242 cau
  -p src/forecasters/hybrid_acd_forecaster.py::HybridACDForecaster
  -o model=<MODEL>
  --num_lines -1      <- QUAN TRONG: -1 de chay toan bo 242 cau
  --run --async
  --output_dir <output>

  -> num_lines=-1 -> num_lines=None (dong 108-109 ground_truth_run.py)

[STEP 2] Doc 242 ForecastingQuestion tu JSONL
  Moi cau co: title, body, resolution_date, resolution=True/False

[STEP 3] Voi moi cau hoi fq:
  fq_no_resolution = pre_call(fq)  <- xoa resolution
  forecast = HybridACDForecaster.call(fq_no_resolution)
    -> Khong co tuple context -> bounds = [0.0, 1.0]
    -> call_with_tcd_sync(fq, 0.0, 1.0) -> van qua CoT + extract

[STEP 4] Tinh Diem
  brier_score = (forecast.prob - resolution)^2
  log_score = -log2(prob + 1e-8)
  brier_score_scaled = 100 * (0.25 - bs) / 0.25

[STEP 5] Luu Ket qua
<output>/
+-- ground_truth_results.jsonl  Moi dong: {question, forecast, prob, resolution, brier_score}
+-- ground_truth_summary.json   {avg_brier_score, calibration_error, decomposition, config}
+-- calibration_plot_linear.png Calibration curve: predicted prob vs. actual frequency
```

---

## 4. Luong Du lieu Toan Bo He thong

```
INPUT DATA
+-- src/data/fq/real/20240501_20240815.jsonl
|   [242 ForecastingQuestion voi resolution=True/False]
|                                    |
|                                    v
|                             Ground Truth Pipeline
|                             -> ground_truth_summary.json
|                                (Brier Score, Calibration Error)
|
+-- src/data/tuples/scraped/
    +-- NegChecker.jsonl             |
    +-- AndChecker.jsonl             |
    +-- OrChecker.jsonl     (10 file)|
    +-- CondChecker.jsonl            |
    +-- ...                          v
                             Consistency Pipeline
                             -> stats_aggregated.json
                                (avg_violation / AVS)
                                     |
                                     v
                             KET QUA - Recent_evaluation/
                             +-- genimi_2.5_flash/
                             |   +-- HybridACD_consistency/ -> stats_aggregated.json
                             |   +-- HybridACD_groundtruth/ -> ground_truth_summary.json
                             |   +-- Basic_consistency/     -> stats_aggregated.json
                             |   +-- Basic_groundtruth/     -> ground_truth_summary.json
                             +-- gpt_4o_mini/  (tuong tu)
                             +-- mistral_medium_3.5/ (tuong tu)
                             +-- mistral_small_4/    (tuong tu)
                             +-- Minimax_M3/         (tuong tu)
                             +-- gpt_5.4_mini/       (nho, thu nghiem)
```

---

## 5. Bang Tong hop Uu tien File

| Uu tien | File | Metric anh huong | Cach anh huong |
|---|---|---|---|
| 5/5 | forecasters/hybrid_acd_forecaster.py | AVS, Brier Score | Core HybridACD logic |
| 5/5 | evaluation.py | AVS | Entry point, tong hop stats |
| 5/5 | ground_truth_run.py | Brier Score | Entry point, tinh diem GT |
| 5/5 | static_checks/Checker.py | AVS | 10 rules + LMSR scoring |
| 5/5 | common/datatypes.py | Tat ca | Data model nen tang |
| 4/5 | common/llm_utils.py | Tat ca | API calls + logit_bias support |
| 4/5 | common/perscache.py | Reproducibility | Cache LLM responses |
| 3/5 | evaluation_utils/proper_scoring.py | Brier, Calibration | Tinh chinh xac scores |
| 3/5 | evaluation_utils/common_options.py | Tat ca | --num_lines -1 bug |
| 3/5 | forecasters/create.py | Tat ca | Factory khoi tao forecaster |
| 3/5 | static_checks/MiniInstantiator.py | Chat luong tuples | Sinh cau hoi bien the |
| 3/5 | static_checks/checker_prompts.py | Chat luong tuples | Prompt templates |
| 2/5 | forecaster_metrics.py | Visualization | Cap dir cho bieu do |
| 2/5 | plot_consistency_vs_brier.py | Visualization | Scatter plot |
| 2/5 | results_analysis.ipynb | Visualization | Phan tich tuong quan |

---

## 6. Tham so va Bien so Quan trong

### Tham so HybridACDForecaster

| Tham so | Mac dinh | Anh huong |
|---|---|---|
| adversarial_enabled | True | Bat Module 1. Tat -> AVS tang 71% |
| tcd_enabled | True | Bat Module 3. Tat -> mat hard guarantee |
| temperature (adv) | 0.7 | Cao = da dang ngon ngu; thap = on dinh |
| logit_bias_value | -100 | Hard block. Giam -> giam calibration drift Mistral |
| parsing_model | gpt-4o-mini-2024-07-18 | Model extract xac suat tu CoT |
| adversarial_model | None (= model) | LLM phu cho rewriting |

### Tham so CLI Quan trong

| Tham so | Gia tri | He qua |
|---|---|---|
| --num_lines | 3 (BUG!) vs -1 | 3 cau = vo nghia thong ke |
| --async | True | Nhanh hon ~20 lan |
| --tuple_dir | scraped vs newsapi | Newsapi co NEGATION anomaly (r=-0.31) |
| -k | all vs subset | Chon subset checkers |

### Environment Variables (.env)

| Variable | Anh huong |
|---|---|
| OPENAI_API_KEY | Bat buoc (GPT + TCD parsing) |
| MISTRAL_API_KEY / ANTHROPIC_API_KEY | Cho cac model tuong ung |
| OPENROUTER_API_KEY | Gemini, MiniMax qua OpenRouter |
| NO_CACHE=True | Tat cache -> random |
| LOCAL_CACHE=True | Local file cache (khong can Redis) |
| MAX_CONCURRENT_QUERIES=25 | Semaphore API calls |
| SIMULATE=True | Test mode, khong goi API |

---

## 7. Diem Yeu Ky thuat Can Lu y

| Van de | File | Muc do | Giai phap |
|---|---|---|---|
| num_lines default=3 | common_options.py:79 | CAO | Luon dung --num_lines -1 |
| Single-token logit bias | hybrid_acd_forecaster.py:315 | TB | Multi-token (0.57) khong duoc mask |
| parsing_model hardcoded | hybrid_acd_forecaster.py:207 | TB | Chỉ dung gpt-4o-mini-2024-07-18 |
| Sequential elicit | hybrid_acd_forecaster.py:88 | Thiet ke | Khong song song duoc, cham hon 1/k |
| Calibration drift Mistral | hybrid_acd_forecaster.py:294 | THAP | logit_bias=-100 qua cung |
| OpenRouter rate limits | llm_utils.py | TB | Gemini/MiniMax qua OpenRouter bi rate limit |

---

## 8. So do Phu thuoc File

```
.env
  |
  v
common/llm_utils.py <--> common/perscache.py
  |
  +-- Duoc import boi:
      |
      +-- common/datatypes.py (data models)
      |
      +-- static_checks/Checker.py
      |   +-- static_checks/MiniInstantiator.py
      |   +-- static_checks/checker_prompts.py
      |
      +-- forecasters/forecaster.py (abstract)
          +-- forecasters/basic_forecaster.py
          +-- forecasters/cot_forecaster.py
          +-- forecasters/hybrid_acd_forecaster.py [CAI TIEN]
          +-- forecasters/consistent_forecaster.py
          +-- forecasters/create.py (factory)
              |
              +-- evaluation_utils/common_options.py (CLI)
              |
              +-- evaluation.py [ENTRY 1] -> stats_aggregated.json
              |
              +-- ground_truth_run.py [ENTRY 2] -> ground_truth_summary.json
                      |
                      +-- evaluation_utils/proper_scoring.py
```

---

*Tai lieu duoc tong hop tu phan tich toan bo 27 file Python va ket qua thuc nghiem.*
*Ngay: 25/06/2026 - Cap nhat khi co thay doi trong: hybrid_acd_forecaster.py, evaluation.py, ground_truth_run.py*
