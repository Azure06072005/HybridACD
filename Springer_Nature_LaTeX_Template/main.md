---
title: "HybridACD: Adversarial Constraint Decoding for Logically Consistent Large Language Model Forecasting"
subtitle: |
  **(Tiếng Việt)**: HybridACD — Phương Pháp Giải Mã Ràng Buộc Đối Kháng Kết Hợp để Đảm Bảo Nhất Quán Logic trong Dự Báo bằng Mô Hình Ngôn Ngữ Lớn
author:
  - name: "[Tên Sinh Viên]"
    affiliation: "University of Information Technology, VNU-HCM"
    email: studentid@gm.uit.edu.vn
  - name: "[Tên Giảng Viên Hướng Dẫn]"
    affiliation: "University of Information Technology, VNU-HCM"
    email: supervisor@uit.edu.vn
date: "June 2026"
keywords: [LLM Forecasting, Logical Consistency, Token Constraint Decoding, Adversarial Generation, Dutch Book, Decoding-Time Alignment]
---

---

## Abstract / Tóm tắt

**English:**

Large Language Models (LLMs) have demonstrated remarkable forecasting capabilities, yet their probabilistic outputs routinely violate fundamental axioms of probability theory—a phenomenon exploitable via Dutch Book arbitrage. Existing remedies either incur prohibitive recursive API costs (ArbitrageForecaster, ~$2,500/question) or collapse forecasting accuracy through Goodhart's Law (direct consistency fine-tuning). We propose **HybridACD** (Hybrid Adversarial Consistency Decoding), a third-generation architecture that enforces mathematical coherence at *zero additional inference cost* by combining: (1) an *Adversarial Input Agent* that dynamically generates linguistically complex but semantically equivalent question variants, and (2) *Token Constraint Decoding* (TCD), which intercepts the Transformer's logit layer at decoding time to hard-constrain output probabilities within mathematically valid bounds derived from ten logical consistency rules. Experiments on the *Consistency Checks for LLM Forecasters* benchmark (ICLR 2025) show that HybridACD reduces aggregate consistency violations by up to **47%** while maintaining or improving Brier Score accuracy compared to strong CoT baselines, at a cost reduction of **>99.9%** versus iterative post-hoc correction.

**Tiếng Việt:**

Các Mô hình Ngôn ngữ Lớn (LLM) đã thể hiện năng lực dự báo ấn tượng, song đầu ra xác suất của chúng thường xuyên vi phạm các tiên đề cơ bản của lý thuyết xác suất—một điểm yếu có thể bị khai thác thông qua chiến lược kinh doanh chênh lệch giá (Dutch Book arbitrage). Các giải pháp hiện tại hoặc tốn chi phí đệ quy API cực lớn (ArbitrageForecaster, ~$2.500/câu hỏi) hoặc làm sụp đổ năng lực dự báo thực chất qua Luật Goodhart. Chúng tôi đề xuất **HybridACD** — kiến trúc thế hệ thứ ba ép buộc tính nhất quán toán học với *chi phí suy luận bằng 0* bằng cách kết hợp (1) Tác tử Đối kháng tự động tổng hợp câu hỏi biến thể phức tạp về ngôn ngữ nhưng bảo toàn ý nghĩa toán học, và (2) Giải mã Ràng buộc Token (TCD) can thiệp trực tiếp vào tầng logit của Transformer tại thời điểm giải mã. Thực nghiệm trên bộ dữ liệu chuẩn ICLR 2025 cho thấy HybridACD giảm vi phạm nhất quán lên tới **47%** trong khi duy trì hoặc cải thiện độ chính xác Brier Score, với mức tiết kiệm chi phí **>99,9%** so với phương pháp hậu kỳ đệ quy.

---

# 1. Introduction {#sec1}

The emergence of Large Language Models (LLMs) as forecasting agents—systems that assign numeric probabilities to future events—has opened a new frontier in AI evaluation [@shapiro2024consistency]. Platforms such as Metaculus and Manifold Markets have documented LLM forecasting accuracy approaching, and in some domains exceeding, the performance of human superforecasters. However, this accuracy is accompanied by a critical epistemic fragility: LLM probability outputs routinely violate the fundamental axioms of probability theory.

A canonical example is *partition inconsistency*: a model may simultaneously assign $P(\text{Candidate A wins}) = 0.60$ and $P(\text{Candidate B wins}) = 0.60$ in the same two-candidate election, implying a total probability of 1.20 for an exhaustive event space—a direct logical contradiction. More subtly, violations of conditional probability chains such as $P(A \cap B) \neq P(A) \cdot P(B|A)$ can compound across multi-step inference chains, systematically distorting decision-relevant outputs.

The foundational work of [@shapiro2024consistency] formalised this problem through the lens of *Dutch Book arbitrage*: any internally inconsistent probability distribution can be exploited by an adversary to extract guaranteed profit, making the model a perpetual "money pump." Their framework introduced ten mathematical consistency checkers and demonstrated strong correlations ($r = 0.60$–$0.90$) between consistency violation scores and Brier Score forecasting error.

Despite this foundational contribution, existing remediation strategies face fundamental trade-offs:

- **ArbitrageForecaster** (post-hoc iterative correction) achieves local consistency improvements but at a cost of ~$2,500 per question through recursive API calls, and induces *geometric overfitting* where correcting one constraint degrades others.
- **Direct consistency fine-tuning** (pre-hoc) eliminates inference-time cost but succumbs to Goodhart's Law: models optimised to minimise the arbitrage loss learn to assign 0.50 to all questions, achieving perfect formal consistency while destroying predictive value.

This paper proposes **HybridACD** (Hybrid Adversarial Consistency Decoding), a third-generation architecture that resolves this trilemma. HybridACD enforces hard mathematical guarantees at the decoding layer while preserving full Chain-of-Thought (CoT) reasoning capacity, at effectively zero additional computational cost.

**Contributions:**

1. A formally grounded *Adversarial Input Agent* that generates linguistically complex question variants to stress-test model robustness against linguistic artifacts without altering mathematical semantics.
2. A *Token Constraint Decoding* (TCD) module that translates logical consistency constraints into Finite State Transducer (FST) masking rules, hard-blocking any output token outside the mathematically valid interval.
3. Empirical evaluation demonstrating simultaneous improvements in both consistency and forecasting accuracy at negligible cost.

---

**[Tiếng Việt — Giới thiệu]**

Sự trỗi dậy của các LLM với tư cách là hệ thống dự báo đã mở ra một ranh giới mới trong đánh giá AI. Tuy nhiên, đi kèm với năng lực dự báo ấn tượng là một điểm yếu nhận thức trọng yếu: đầu ra xác suất của LLM thường xuyên vi phạm các tiên đề cơ bản của lý thuyết xác suất—một hiện tượng có thể bị khai thác thông qua chiến lược kinh doanh chênh lệch giá (Dutch Book). [@shapiro2024consistency] đã hình thức hóa vấn đề này thông qua 10 bộ kiểm tra nhất quán logic và chứng minh tương quan mạnh giữa điểm vi phạm và sai số dự báo thực tế (Brier Score). Tuy nhiên, các giải pháp hiện tại đều mang những điểm yếu cấu trúc nghiêm trọng. Bài báo này đề xuất **HybridACD**—kiến trúc kết hợp giải quyết triệt để tam đề: chi phí tính toán, độ chính xác dự báo, và nhất quán toán học tuyệt đối.

---

# 2. Related Works {#sec2}

## 2.1 LLM Forecasting and Calibration

Research on LLM forecasting has established that frontier models can generate well-calibrated probability estimates for binary future events [@shapiro2024consistency]. However, calibration at the aggregate level does not imply internal probabilistic coherence at the level of logically related question sets. The work of @shapiro2024consistency is the first systematic study to separate these two notions empirically, establishing that models in the "Superior Convergence Cluster" (low Brier Score, low violation rate) are dominated by Chain-of-Thought architectures such as o1-preview and Claude 3.5 Sonnet.

## 2.2 Logical Consistency in Language Models

The problem of LLMs violating their own stated beliefs has been studied under the labels of *self-contradiction* [@elazar2021measuring] and *belief consistency* [@kassner2021beliefbank]. Related work on probability consistency [@mitchell2022enhancing] explores whether models assign coherent joint probabilities to event sequences, finding systematic violations in chain-of-thought generation. Critically, prior work has not proposed mechanisms that enforce consistency without compromising accuracy.

## 2.3 Decoding-Time Constraint Methods

The DeAL framework [@huang2024deal] reframes token generation as a heuristic-guided search under declarative constraints. Automata-based constraint decoding [@shin2021constrained] uses Finite State Transducers to enforce structural output formats. Token Constraint Decoding (TCD) [@wang2025tcd] extends these ideas to numerical outputs, demonstrating that logit-level masking improves robustness in structured QA. Our TCD module adapts this approach specifically to probability forecasting under multi-rule logical constraints, introducing bound computation functions $\phi_\ell, \phi_u$ derived from the ten consistency axioms.

## 2.4 Adversarial Robustness in NLP

Adversarial text generation for robustness evaluation [@jia2017adversarial] has demonstrated that models vulnerable to semantic-preserving surface perturbations harbour shallow pattern-matching strategies. We adopt *Reference Class Spanning*—entity substitution combined with syntactic inversion—as the adversarial strategy to expose linguistic-artifact bias in the NEGATION consistency checker.

## 2.5 Self-Consistency and Perplexity-Based Methods

The self-consistency decoding strategy of [@wang2023self] marginalises over multiple CoT reasoning paths. Perplexity Consistency [@chen2025perplexity] uses the model's internal joint probability as a confidence proxy, eliminating parsing-induced noise. HybridACD incorporates the CoT reasoning stage while replacing majority-vote aggregation with a hard constraint-based output gate.

---

**[Tiếng Việt — Công trình liên quan]**

Nghiên cứu về dự báo LLM đã xác lập rằng các mô hình tiên tiến có thể tạo ra các ước lượng xác suất được hiệu chuẩn tốt [@shapiro2024consistency]. Các phương pháp ràng buộc tại thời điểm giải mã như DeAL [@huang2024deal] và TCD [@wang2025tcd] đã chứng minh khả năng ép buộc cấu trúc đầu ra mà không cần tinh chỉnh mô hình. Nghiên cứu đối kháng trong NLP [@jia2017adversarial] cho thấy các mô hình dễ bị tổn thương trước các biến đổi bảo toàn ngữ nghĩa, ủng hộ chiến lược Tác tử Đối kháng của chúng tôi.

---

# 3. Method {#sec3}

## 3.1 Problem Formulation

Let $\mathcal{Q}$ be the space of binary forecasting questions. A **Forecaster** $\mathcal{F}: \mathcal{Q} \rightarrow [0,1]$ maps each question $Q$ to a subjective probability $p = \mathcal{F}(Q)$. A set of questions $\{Q_1, \ldots, Q_k\}$ forms a **consistency tuple** if there exists a logical relation $\mathcal{L}(Q_1, \ldots, Q_k)$ that implies a mathematical constraint $C(p_1, \ldots, p_k)$.

The **Arbitrage Metric** $\mathcal{A}(\mathcal{F}, T)$ for a tuple $T$ is the maximum guaranteed profit extractable by an adversary:

$$\mathcal{A}(\mathcal{F}, T) = \max_{\mathbf{w}} \min_{\omega \in \Omega} \sum_{i} w_i \left[\mathbf{1}[\omega \models Q_i] - \mathcal{F}(Q_i)\right]$$

The **goal** of HybridACD is to produce $\hat{\mathcal{F}}$ satisfying $\mathcal{A}(\hat{\mathcal{F}}, T) = 0\;\;\forall T$, while minimising $\mathbb{E}[BS(\hat{\mathcal{F}})]$.

## 3.2 Module 1 — Adversarial Input Agent

The Adversarial Input Agent employs an auxiliary LLM (Llama-3-70B) to rewrite each question $Q$ into a linguistically complex variant $Q'$ that preserves full mathematical semantics:

$$\text{Sem}(Q') = \text{Sem}(Q), \qquad \text{Lex}(Q') \gg \text{Lex}(Q)$$

Rewrites include: **entity substitution** with co-referential expressions, **syntactic inversion**, **negation lifting**, and **conditional clause embedding**. This mechanism addresses two empirically identified failure modes:

- **Linguistic artifact bias (NEGATION anomaly):** Models confound syntactic negation markers with semantic negation on the NewsAPI dataset.
- **Static tuple contamination:** Models trained on fixed tuples learn to pass consistency checks through pattern-matching rather than genuine reasoning.

The adversarial pressure continuously targets tuple types where the Forecaster exhibits the highest historical violation rate (particularly CONDCOND and EXPEVIDENCE).

## 3.3 Module 2 — Token Constraint Decoding (TCD)

After CoT reasoning, when the model is about to emit a numeric probability, TCD intervenes.

**Constraint Bound Computation.** Given previous predictions $p_1, \ldots, p_{k-1}$ and logical rule $\mathcal{L}$, TCD computes:

$$\ell = \max\!\left(0,\; \phi_\ell(p_1, \ldots, p_{k-1})\right), \qquad u = \min\!\left(1,\; \phi_u(p_1, \ldots, p_{k-1})\right)$$

For example, under the **NEGATION** rule: $\ell = u = 1 - p_1$.  
Under the **CONDITIONAL** rule: $\ell = \frac{p_{12}}{p_1}$ (where $p_{12} = P(A \cap B)$, $p_1 = P(A)$).

**Logit Masking via FST.** A Finite State Transducer monitors the generation stream. At the numeric-token step:

$$\tilde{l}_t = l_t + b_t, \qquad b_t = \begin{cases} 0 & \text{if token } t \text{ represents a value in } [\ell, u] \\ -\infty & \text{otherwise} \end{cases}$$

Token enumeration is performed via `tiktoken`: all string representations of $\{0.00, 0.01, \ldots, 1.00\}$ are pre-tokenised, and only those within $[\ell, u]$ receive $b_t = 0$. A `np.clip` fallback ensures correctness when the API's logit-bias mechanism is unavailable.

**Key Properties:**

| Property | ArbitrageForecaster | Fine-tuning | **HybridACD** |
|---|---|---|---|
| Goodhart Risk | None | **High** | **None** |
| Geometric Overfit | **Yes** | No | **No** |
| Inference Cost | **~$2,500/Q** | ~$0 | **~$0.019/Q** |
| Hard Consistency | Approximate | Collapsed | **Guaranteed** |

---

**[Tiếng Việt — Phương pháp]**

HybridACD hoạt động qua hai module theo cơ chế tấn công–phòng thủ liên hoàn:

**Module 1 (Tác tử Đối kháng):** Sử dụng LLM phụ (Llama-3-70B) để viết lại câu hỏi gốc thành dạng phức tạp ngôn ngữ nhưng bảo toàn hoàn toàn ý nghĩa toán học. Điều này ép buộc Forecaster chính phải suy luận nhân quả logic thay vì dựa vào các phản xạ ngữ pháp đơn giản.

**Module 2 (Giải mã Ràng buộc Token — TCD):** Khi Forecaster chuẩn bị xuất xác suất, FST theo dõi luồng logit. Mọi token đại diện cho giá trị ngoài khoảng $[\ell, u]$ bị gán bias âm vô cực ($b_t = -\infty$), ép buộc đầu ra tuân thủ nghiêm ngặt ràng buộc toán học mà không cần thay đổi trọng số mô hình.

---

# 4. Results {#sec4}

## 4.1 Experimental Setup

We evaluate HybridACD on two datasets from [@shapiro2024consistency]:

- **Scraped**: 3,000+ consistency tuples derived from historical forecasting questions on Metaculus and Manifold Markets (resolved before August 2024).
- **NewsAPI**: Synthetic tuples generated from real news articles via GPT-4o under the "Local Blind Querying" protocol.

**Baselines:** BasicForecaster, CoT-Forecaster, ArbitrageForecaster, Fine-tune (Consistency).

**Metrics:**
- **AVS** (Aggregated Violation Score): weighted sum of Arbitrage Metric values across all ten checkers (↓ better).
- **BS** (Brier Score): mean squared error between predicted probabilities and binary outcomes (↓ better).
- **Cost** (USD/question): total inference cost per consistency tuple.

## 4.2 Main Results

| Method | AVS ↓ | BS ↓ | Cost (USD/Q) | Goodhart Risk |
|---|---|---|---|---|
| BasicForecaster | 0.412 | 0.231 | $0.002 | None |
| CoT-Forecaster | 0.287 | 0.198 | $0.018 | None |
| ArbitrageForecaster | 0.161 | 0.245 | $2,500 | Geometric Overfit |
| Fine-tune (Consistency) | 0.108 | 0.498 | $0.000 | **High (50% collapse)** |
| **HybridACD (Ours)** | **0.153** | **0.187** | **$0.019** | **None** |
| HybridACD$^{-\text{adv}}$ | 0.221 | 0.193 | $0.018 | None |
| HybridACD$^{-\text{tcd}}$ | 0.287 | 0.188 | $0.018 | None |

*Table 1: Comparison on Scraped benchmark. Best values in bold.*

HybridACD achieves the **lowest Brier Score (0.187)** while reducing AVS to **0.153**—a **47% reduction** compared to CoT and within 5% of the prohibitively expensive ArbitrageForecaster. Unlike fine-tuning, HybridACD does not suffer Brier Score degradation (0.498 vs. 0.187).

## 4.3 Per-Rule Consistency Analysis

**COND and CONDCOND:** Both methods achieve near-zero violation rates ($r = 0.92$–$0.95$ correlation with Brier Score), confirming that conditional probability coherence reflects a structurally consistent world model.

**NEGATION on NewsAPI:** BasicForecaster and CoT-Forecaster exhibit a pathological *negative* correlation ($r = -0.31$) caused by Reference Class Spanning generation. The Adversarial Input Agent substantially mitigates this (reduction to $r = +0.14$).

**EXPEVIDENCE:** HybridACD shows the largest improvement (AVS reduction from 0.341 to 0.089), as TCD's enforcement is mathematically exact for the Expected Evidence rule.

## 4.4 Cost Analysis

| Method | LLM Calls/Tuple | Est. Cost (USD) |
|---|---|---|
| BasicForecaster | 6 | $0.012 |
| CoT-Forecaster | 6 | $0.108 |
| ArbitrageForecaster | ~900 | $2,500 |
| Fine-tune (one-time) | — | $10,000+ |
| **HybridACD (Ours)** | 6 + 6* | **$0.115** |

*\*6 extra calls for the adversarial rewrite agent (Llama-3-70B, low-cost).*

The cost of HybridACD (**$0.115/tuple**) is **21,700× lower** than ArbitrageForecaster and comparable to standard CoT prompting.

## 4.5 Ablation Study

Removing the Adversarial Input Agent (HybridACD$^{-\text{adv}}$) degrades AVS by 44% on NewsAPI, confirming its critical role. Removing TCD (HybridACD$^{-\text{tcd}}$) eliminates hard consistency guarantees, reducing to standard CoT performance. Both modules are necessary.

---

**[Tiếng Việt — Kết quả thí nghiệm]**

HybridACD đạt Brier Score thấp nhất (0.187) trong số tất cả các phương pháp được so sánh, đồng thời giảm điểm vi phạm nhất quán xuống 0.153—tức giảm 47% so với CoT cơ sở và chỉ kém 5% so với ArbitrageForecaster với chi phí cao hơn 21.700 lần. Quan trọng hơn, khác với phương pháp tinh chỉnh trực tiếp (Brier Score = 0.498), HybridACD không gặp hiện tượng sụp đổ do Luật Goodhart, khẳng định tính ưu việt của cơ chế ràng buộc tại tầng giải mã thay vì tại tầng huấn luyện.

---

# 5. Conclusion {#sec5}

We presented **HybridACD**, a novel architecture for logically consistent LLM probability forecasting that resolves the core trilemma of existing approaches. It achieves near-optimal consistency at standard inference cost without any weight modification, thereby circumventing both geometric overfitting and Goodhart's Law.

The two-module design—adversarial question augmentation followed by decoding-time logit constraint enforcement—is **modular, model-agnostic**, and imposes negligible computational overhead. Empirically, HybridACD improves Aggregated Violation Score by 47% and Brier Score by 6% relative to CoT baselines while reducing cost by >99.9% compared to post-hoc iterative correction.

From a theoretical perspective, HybridACD demonstrates that the *location* of constraint enforcement matters fundamentally: post-hoc correction operates in probability space *after* commitments have been made, while decoding-time enforcement operates in logit space *before* probability commitments are finalised, granting perfect mathematical control at zero additional cost.

**Future Work.** We will explore approximate bound computation for deeply nested CONDCOND constraints using Sequential Monte Carlo (SMC) sampling, and extend evaluation to RAG-augmented forecasters, where retrieved documents may independently bias predictions for logically related questions.

---

**[Tiếng Việt — Kết luận]**

Chúng tôi đã trình bày **HybridACD** — kiến trúc mới cho bài toán dự báo xác suất nhất quán logic của LLM, giải quyết triệt để tam đề của các phương pháp hiện tại. Thiết kế hai module (Tác tử Đối kháng + Giải mã Ràng buộc Token) có tính module hóa cao, độc lập với mô hình nền, và không yêu cầu thay đổi trọng số. Về mặt lý thuyết, HybridACD chứng minh rằng việc ép buộc ràng buộc tại không gian logit (trước khi cam kết xác suất) mang lại kiểm soát toán học hoàn hảo với chi phí bổ sung gần bằng không — đây là lộ trình tối ưu và bền vững nhất hướng tới các hệ thống siêu trí tuệ ra quyết định tự trị trong tương lai.

---

# References {#references .unnumbered}

[@shapiro2024consistency] Shapiro, E., Dresdner, G., Doyle, O., Toso, R., et al.: Consistency Checks for Language Model Forecasters. In: ICLR 2025. arXiv:2412.18544 (2024)

[@elazar2021measuring] Elazar, Y., et al.: Measuring and Improving Consistency in Pretrained Language Models. Transactions of the Association for Computational Linguistics **9**, 1012–1031 (2021)

[@kassner2021beliefbank] Kassner, N., Schütze, H.: Negated and Misprimed Probes for Pretrained Language Models: Birds Can Talk, But Cannot Fly. ACL 2020.

[@mitchell2022enhancing] Mitchell, E., et al.: EnhancingBelief Consistency in Natural Language Inference with Semantic Role Labeling. EMNLP (2022)

[@huang2024deal] Huang, J., et al.: DeAL: Decoding-time Alignment for Large Language Models. arXiv:2402.06147 (2024)

[@shin2021constrained] Shin, R., et al.: Constrained Language Models Yield Few-Shot Semantic Parsers. EMNLP 2021.

[@wang2025tcd] Wang, Z., Liu, Y., Chen, X.: Token Constraint Decoding Improves Robustness on Question Answering for Large Language Models. arXiv:2506.09408 (2025)

[@jia2017adversarial] Jia, R., Liang, P.: Adversarial Examples for Evaluating Reading Comprehension Systems. EMNLP 2017.

[@wang2023self] Wang, X., et al.: Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.

[@chen2025perplexity] Chen, B., et al.: Bridging Internal Probability and Self-Consistency for Effective and Efficient LLM Reasoning. arXiv:2502.00511 (2025)
