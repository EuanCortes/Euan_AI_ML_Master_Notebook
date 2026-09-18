# The Applied AI Engineer's Compendium: Master Checklist

This document is the **single source of truth** for the curriculum. It defines the
nine chapters of the theory book and the chapter-to-project mapping used
throughout the repository. `docs/ProjectPlanOverview.md` covers architecture and
execution strategy and links back here rather than repeating this content.

The curriculum runs in three phases: build the mathematical and engineering
foundations, work through applied ML and AI, then ship the whole thing as a
production system.

| Phase | Chapter | Title | Code |
| :--- | :--- | :--- | :--- |
| 1 | 1 | Mathematical Foundations for Machine Learning | — |
| 1 | 2 | Engineering Foundations & Infrastructure | `projects/02_engineering_foundations/` |
| 2 | 3 | Data, Classical ML & Model Evaluation | `projects/03_classical_ml/` |
| 2 | 4 | Bayesian ML & Probabilistic Modeling | `projects/04_bayesian_ml/` |
| 2 | 5 | Time Series Analysis & Financial Engineering | `projects/05_time_series/` |
| 2 | 6 | Deep Learning Fundamentals | `projects/06_deep_learning/` |
| 2 | 7 | Computer Vision & Scientific ML | `projects/07_cv_sciml/` |
| 2 | 8 | Generative AI, NLP & Agentic Systems | `projects/08_gen_ai/` |
| 3 | 9 | End-to-End Systems Design & MLOps (Capstone) | `projects/09_capstone/` |

Each chapter carries two halves. **Theory** is the mathematics and the design
principles, written in LaTeX under `theory/chapters/`. **Applied** is the
executable implementation under `projects/`.

**The stack is Python end to end.** Data work is pandas/Polars, not SQL;
persistence is Python-native file formats, not a relational database. That is a
deliberate scope decision — it keeps every chapter in one language.

---

# Phase 1: Mathematical & Engineering Foundations

Chapter 1 gives you the language the models are written in. Chapter 2 gives you
the machinery to build them reproducibly — done up front so every later chapter
inherits it rather than retrofitting it.

## Chapter 1: Mathematical Foundations for Machine Learning

*Before you can build the models, you need to speak the language they are written in.*

The three pillars of this chapter are **Linear Algebra**, **Matrix Calculus**, and
**Multivariate Optimization**. Probability, statistical inference, information
theory, and numerical computing follow as supporting sections — they are
prerequisites for Chapters 3, 4, and 8 rather than optional extras.

*   **Linear Algebra:** Vectors, matrices, tensors, dot products, eigenvalue/eigenvector decomposition, Singular Value Decomposition (SVD) ($A = U \Sigma V^T$).
*   **Matrix Calculus:** Partial derivatives, the chain rule (the heart of backpropagation), gradients, Jacobians, Hessians, Taylor series expansion, and the layout conventions for differentiating vector- and matrix-valued functions.
*   **Multivariate Optimization:**
    *   Convexity, stationary points, and second-order conditions.
    *   Constrained optimization: Lagrange multipliers and the KKT conditions.
    *   Gradient descent as an optimization scheme (the algorithmic variants land in Chapter 6).
*   **Numerical Computing & Stability:**
    *   Floating-point representation (FP64/FP32), machine epsilon, and rounding error.
    *   Catastrophic cancellation and loss of significance.
    *   The log-sum-exp trick, and why probabilities are carried in log space.
    *   Matrix condition number; why you *solve* a linear system rather than invert the matrix.
*   **Probability & Statistics:**
    *   Probability distributions (Normal, Poisson, Bernoulli, Beta, Dirichlet).
    *   Bayes' Theorem, Maximum Likelihood Estimation (MLE), Maximum A Posteriori (MAP).
    *   Expected value, variance, covariance matrices.
*   **Statistical Inference & Experiment Design:**
    *   Sampling distributions, the Central Limit Theorem, standard error.
    *   Confidence intervals; the bootstrap.
    *   Hypothesis testing (t-test, chi-squared, Mann-Whitney), p-values and how they are misread.
    *   Multiple-comparison correction (Bonferroni, Benjamini-Hochberg).
    *   Statistical power, effect size, and sample-size calculation.
    *   A/B testing: randomisation, sequential-testing pitfalls, and answering "is model B actually better than model A?"
*   **Information Theory:** Entropy ($H(X)$), Cross-Entropy, Kullback-Leibler (KL) Divergence ($D_{KL}(P||Q)$).

## Chapter 2: Engineering Foundations & Infrastructure (The "How")

*The discipline that keeps the repository scalable. Build this before the models, not after.*

### Theory

*   **Systems Design Principles:** Separation of concerns, dependency direction, idempotency, and the reproducibility contract (same inputs, same commit, same outputs).
*   **Monorepo Architecture:** Why theory, library code, and projects live in one repository; shared-library versus per-project dependency boundaries; the cost model of a monorepo versus many small repositories.
*   **Isolated Environments:** Virtual environments, lockfiles versus loose version ranges, and why a global `requirements.txt` fails as soon as two projects disagree on a version.

### Applied

*   **Dependency Management with `uv`:** Project initialisation, `uv sync` / `uv add` / `uv run`, resolving and committing `uv.lock`, pinning the interpreter via `.python-version`, and per-project `pyproject.toml` files for isolation.
*   **Data Version Control (DVC):** Initialising DVC, tracking datasets as lightweight pointers, configuring a remote (AWS S3 / Google Drive), and the `dvc pull` / `dvc push` round trip. Large datasets never enter Git history.
*   **Git Branch Protection:** Trunk-based workflow on `main`, feature branches, required pull-request review, required passing status checks, and a linear history with no force-pushes to `main`.
*   **CI/CD via GitHub Actions:** Workflow anatomy (triggers, jobs, steps), path filters so a LaTeX edit does not run the Python suite, automated PDF compilation of the theory book, `pytest` on every pull request, linting and formatting gates, and artifact upload.
*   **Core Software Engineering:** Unit testing with `pytest`, fixtures and parametrisation, type hints, and docstrings as the baseline standard for every module in `src/` and `projects/`.

---

# Phase 2: Applied Machine Learning & AI

Chapters 3 through 8 progress from classical machine learning to generative AI.
Each chapter derives the mathematics, then builds the actual model.

## Chapter 3: Data, Classical ML & Model Evaluation

*The first applied chapter. Get the data right, fit the models that still win on tabular problems, then learn to tell whether they work.*

### Data Foundations

*Most modelling time is spent here. Everything is pandas/Polars — the stack stays Python.*

*   **Data Loading & Exploratory Analysis:** pandas and Polars dataframes, vectorised operations instead of Python loops, joins/merges and `groupby` aggregation, profiling an unfamiliar dataset, inspecting distributions and correlations.
*   **Data Cleaning:** Missing-data mechanisms (MCAR/MAR/MNAR) and imputation strategies, outlier detection, duplicate handling, dtype coercion and memory footprint.
*   **Feature Engineering:** Scaling and standardisation, categorical encoding (one-hot, ordinal, target), interaction and polynomial features, binning, and cyclical encoding for temporal features.
*   **Dimensionality Reduction:** PCA as the workhorse — derived from the covariance matrix under *Unsupervised Learning* below — plus t-SNE and UMAP for visualisation only, and when each is the wrong choice.

### Core Models

*   **Linear Models:** Ordinary Least Squares (OLS) closed-form solution ($\beta = (X^TX)^{-1}X^TY$), Logistic Regression (Sigmoid function mapping), Ridge (L2) and Lasso (L1) regularization penalties.
*   **Tree-Based Models:** Decision Trees (Information Gain, Gini impurity equations), Random Forests (bagging math), Gradient Boosting Machines (XGBoost, LightGBM—derive the Taylor expansion of the loss function).
*   **Support Vector Machines (SVM):** The primal and dual optimization problems, the kernel trick (RBF, polynomial), max-margin classification.
*   **Unsupervised Learning:** K-Means clustering (Lloyd's algorithm), Hierarchical clustering, Principal Component Analysis (PCA) (deriving PCA via eigenvalue decomposition of the covariance matrix).
*   **Recommender Systems:** Collaborative versus content-based filtering, matrix factorization via the SVD of Chapter 1, implicit feedback, the cold-start problem, and ranking metrics (MAP@k, NDCG).

### Evaluation, Bias & Lifecycle

*   **The Bias-Variance Tradeoff:** Deriving $Error = Bias^2 + Variance + Irreducible Error$.
*   **Validation Strategies:** K-fold cross-validation, stratified sampling, temporal splits (crucial for time series!).
*   **Metrics:** Confusion matrix, Precision, Recall, F1-Score, ROC-AUC, RMSE, MAE, R-squared.
*   **Hyperparameter Optimization:** Grid and random search, successive halving, Bayesian optimization with Gaussian Processes (the GP machinery arrives in Chapter 4), and `Optuna` in practice.
*   **Model Interpretability:**
    *   **SHAP** — Shapley values from cooperative game theory, TreeSHAP for ensembles, local versus global explanations, summary and force plots. This is the standard expectation when presenting ensemble results, so treat it as core rather than optional.
    *   Permutation importance, partial dependence plots, and LIME.
    *   Why built-in tree feature importances mislead.
*   **Data Issues:** Imbalanced datasets (SMOTE math), data leakage, algorithmic bias and fairness definitions (Demographic Parity, Equalized Odds).
*   **Experiment Tracking:** Weights & Biases (W&B) or MLflow — introduced here and used for every model trained from this chapter onward.

## Chapter 4: Bayesian ML & Probabilistic Modeling

*Focus on how these models represent uncertainty mathematically.*

*   **Core Bayesian Inference:**
    *   Write out the full Bayes integral: $P(\theta|D) = \frac{P(D|\theta)P(\theta)}{\int P(D|\theta)P(\theta)d\theta}$.
    *   Conjugate priors (e.g., how a Beta prior and Binomial likelihood yield a Beta posterior).
*   **Approximate Inference (Solving intractable integrations):**
    *   *Markov Chain Monte Carlo (MCMC):* Detailed balance, Metropolis-Hastings acceptance probability, Hamiltonian Monte Carlo (HMC).
    *   *Variational Inference (VI):* Deriving the Evidence Lower Bound (ELBO): $\mathbb{E}_{q}[\log P(X,Z) - \log q(Z)]$.
*   **Gaussian Processes (GPs):**
    *   The math of defining distributions over functions.
    *   Covariance functions/kernels (RBF, Matern).
    *   Deriving the predictive mean and variance matrices.
*   **Bayesian Deep Learning:** Bayesian Neural Networks (BNNs), Monte Carlo (MC) Dropout (prove how dropout approximates Bayesian inference).

## Chapter 5: Time Series Analysis & Financial Engineering

*Modeling data across time, volatility, and stochastic processes.*

*   **Classical Time Series Statistics:**
    *   Stationarity, Autocorrelation (ACF) and Partial Autocorrelation (PACF).
    *   Augmented Dickey-Fuller (ADF) test.
    *   ARIMA/SARIMA models (Auto-Regressive, Integrated, Moving Average equations).
*   **Volatility Modeling (Financial specific):**
    *   ARCH and GARCH models (modeling conditional variance/heteroskedasticity over time).
*   **Stochastic Calculus & Quantitative Finance:**
    *   Brownian Motion / Wiener Processes ($dW_t$).
    *   Ito's Lemma (the calculus of stochastic processes).
    *   Geometric Brownian Motion (GBM) and deriving the Black-Scholes PDE.
    *   Portfolio Optimization: Markowitz Efficient Frontier (mean-variance optimization).
*   **Deep Time Series:** Temporal Convolutional Networks (TCNs), Time Series Transformers (PatchTST).
*   **Backtesting:** Walk-forward validation, look-ahead bias, survivorship bias.

## Chapter 6: Deep Learning Fundamentals

*   **The Multilayer Perceptron (MLP):** Nodes, weights, biases. Write out the forward pass and backpropagation (chain rule) by hand.
*   **Activation Functions:** Sigmoid, Tanh, ReLU, GELU, Softmax (write their equations and first derivatives).
*   **Optimization Algorithms:** Gradient Descent update rules, Momentum, RMSProp, Adam (derive the bias-corrected first and second moment estimates).
*   **Loss Functions:** MSE, Binary/Categorical Cross-Entropy, Hinge Loss.
*   **Regularization:** Dropout, Batch Normalization (write the mean/variance centering equations), Early Stopping.
*   **Sequence Models:**
    *   Recurrent Neural Networks (RNNs) and Backpropagation Through Time (BPTT).
    *   Vanishing and exploding gradients over time — the concrete failure the gates were invented to fix.
    *   LSTM and GRU gating equations.
    *   Sequence-to-sequence encoder/decoder models and the fixed-context bottleneck.
    *   Additive (Bahdanau) attention — the direct bridge to the Transformer in Chapter 8.
*   **Efficient Training & High-Performance Computing:**
    *   The GPU execution model: SIMT, warps, memory hierarchy, and why batch size drives utilisation.
    *   Memory arithmetic: parameters + activations + gradients + optimizer states, and how to predict whether a model fits.
    *   Mixed precision (FP16/BF16), loss scaling, and numerical caveats that tie back to Chapter 1.
    *   Gradient accumulation and gradient (activation) checkpointing.
    *   Data, model, tensor, and pipeline parallelism; DDP, FSDP, and ZeRO sharding.
    *   Profiling, throughput measurement, and finding the actual bottleneck.
    *   Cluster scheduling (SLURM), multi-GPU jobs, and checkpoint/resume discipline.

## Chapter 7: Computer Vision & Scientific ML

### Computer Vision (CV)

*   **CNNs:** Convolution operations (write the discrete convolution math), pooling layers, strides, padding, receptive fields.
*   **Classic Architectures:** ResNet (skip connection math: $H(x) = F(x) + x$), VGG.
*   **Object Detection & Segmentation:** YOLO, Mask R-CNN, Intersection over Union (IoU) calculation.
*   **Modern CV:** Vision Transformers (ViT) and image patch embedding.

### Scientific Machine Learning (SciML)

*   **Physics-Informed Neural Networks (PINNs):** Embedding partial differential equations (PDEs) directly into the neural network loss function ($Loss = Loss_{data} + Loss_{PDE}$).
*   **Neural Operators:** Fourier Neural Operator (FNO) and solving PDEs in the frequency domain.

## Chapter 8: Generative AI, NLP & Agentic Systems

### Foundational Generative Models

*   **VAEs:** The reparameterization trick ($z = \mu + \sigma \odot \epsilon$).
*   **GANs:** The minimax game loss function.
*   **Diffusion Models (DDPMs):** The Markov chain forward noise equations and reverse denoising U-Net process.

### Large Language Models (LLMs)

*   **The Transformer Architecture:** Write out the exact Scaled Dot-Product Attention equation ($Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$).
*   **Positional Encoding:** sine/cosine equations.
*   **Parameter-Efficient Fine-Tuning (PEFT):** LoRA math (low-rank matrix decomposition $W = W_0 + BA$).
*   **Local Inference & Quantization:** Math behind compression (FP16 to INT8/INT4), KV Cache mechanics.
*   **Hugging Face Ecosystem:** `transformers`, `datasets`, Hub management.

### Reinforcement Learning Fundamentals

*Prerequisite for alignment. RLHF and PPO are unreadable without it.*

*   Markov Decision Processes: states, actions, rewards, transitions, discounting.
*   Value and action-value ($Q$) functions; the Bellman expectation and optimality equations.
*   Dynamic programming: policy iteration and value iteration.
*   Temporal-difference learning, Q-learning, and Deep Q-Networks.
*   Policy gradients: the policy gradient theorem, REINFORCE, and variance reduction via baselines.
*   Actor-critic methods; advantage estimation (GAE).
*   Trust regions and clipped surrogate objectives — the road to TRPO and PPO.

### Alignment

*   Supervised fine-tuning.
*   Reward modeling from preference pairs.
*   RLHF with PPO (building directly on the RL section above).
*   DPO (Direct Preference Optimization loss) and why it removes the RL loop.

### Retrieval & Agentic Systems

*   **Prompt Engineering:** Few-shot, Chain-of-Thought (CoT), Tree-of-Thought.
*   **Retrieval-Augmented Generation (RAG):**
    *   Vector databases (Chroma, FAISS).
    *   Distance metrics: Cosine similarity ($\frac{A \cdot B}{||A|| ||B||}$), Euclidean distance.
    *   Chunking strategies, embedding models, and the full ingestion-to-answer pipeline.
*   **Agentic Systems:** ReAct framework, tool/function calling logic, Directed Acyclic Graphs (DAGs) for multi-agent orchestration (LangGraph).
*   **Generative Evaluation & Benchmarking:**
    *   *N-gram metrics:* BLEU, ROUGE formulas.
    *   *Statistical metrics:* Perplexity ($2^{H(P)}$), Cross-Entropy.
    *   *RAG metrics:* Context precision/recall, faithfulness.

### Adversarial Robustness & AI Security

*Chapter 9 exposes a model over HTTP, which makes this operational rather than theoretical.*

*   Adversarial examples: FGSM and PGD; why small input perturbations flip predictions.
*   Data poisoning and backdoor attacks.
*   Prompt injection, jailbreaks, and indirect injection through retrieved documents.
*   Model extraction and membership-inference attacks.
*   PII handling, data minimisation, and the basics of differential privacy.
*   Input validation and output guardrails.

---

# Phase 3: Production & Deployment

## Chapter 9: End-to-End Systems Design & MLOps (The Capstone)

*The chapter that turns the portfolio into a running system. It consumes a model
trained in an earlier chapter and deploys it as a live, containerised service.*

Persistence here is Python-native — structured logs and columnar files, not a
relational database. That keeps the capstone in one language and the focus on
service design.

### Theory

*   **From Notebooks to Live Services:** Why a Jupyter notebook is not a deployment artifact; the training/serving boundary; model artifacts, versioning, and the training-serving skew problem.
*   **Data Persistence for Services:** What must be durable versus ephemeral, append-only event logs, schema evolution for logged records, and why the serving path should never block on a write.
*   **Microservice Architecture:** Service decomposition, statelessness, synchronous versus asynchronous boundaries, health checks, and the twelve-factor configuration model.

### Applied

*   **RESTful APIs with FastAPI:** REST principles, path and query parameters, Pydantic request/response schemas, dependency injection, wrapping a trained model behind a `/predict` endpoint, and auto-generated OpenAPI documentation.
*   **Prediction Logging in Python:** Append-only JSONL for request/response records, Parquet via `pandas`/`pyarrow` for analysis, partitioning by date, rotation, and reading the log back to compute drift.
*   **Containerization & Orchestration:**
    *   Docker architecture: images versus containers, layer caching, multi-stage builds for slim Python images.
    *   `docker compose` to orchestrate the API and its supporting services as one stack, with volumes, networks, and environment configuration.
*   **Model Serving at Scale:** Inference engines (`vLLM`, `llama.cpp`), batching, and the latency-versus-throughput tradeoff.
*   **Cloud Deployment:** AWS (EC2, S3, SageMaker) as the hosting target for the containerised stack.
*   **Monitoring:** Structured logging, drift detection, and the retraining trigger.
