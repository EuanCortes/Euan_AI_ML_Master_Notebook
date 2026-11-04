# The Applied AI Engineer's Compendium

A self-directed curriculum that takes machine learning from first principles to
production. Each of its nine chapters has two halves: the mathematics, derived by
hand and written up as a LaTeX book in `theory/`, and an applied project in
`projects/` that turns those ideas into tested, reproducible Python. The
engineering infrastructure comes first, as Chapter 2, before any model is
trained: dependency locking with `uv`, data versioning with DVC, CI with GitHub
Actions and pull-request-only merges to `main`. Every later project inherits it.
The stack is Python from end to end. The curriculum finishes with a capstone that
deploys a model from an earlier chapter as a containerised FastAPI service.

## Repository Architecture

```text
Euan_AI_ML_Master_Notebook/
├── theory/                          # LaTeX source for the theory book
│   ├── main.tex                     # Master document — compile this
│   ├── preamble.tex                 # Packages, styling, theorem boxes, macros
│   ├── chapters/                    # One folder per chapter
│   │   └── 0N_<slug>/0N_<slug>.tex
│   ├── figures/                     # Shared images
│   └── references.bib               # Single shared bibliography
├── src/
│   └── mlnotebook/                  # Shared library code, installed via uv
├── projects/                        # Executable code, one folder per chapter
│   ├── 01_mathematical_foundations/
│   ├── 02_engineering_foundations/
│   ├── 03_classical_ml/
│   ├── 04_bayesian_ml/
│   ├── 05_time_series/
│   ├── 06_deep_learning/
│   ├── 07_cv_sciml/
│   ├── 08_gen_ai/
│   └── 09_capstone/
├── .github/workflows/               # CI/CD pipelines
│   └── compile_theory.yml           # Builds the theory book PDF
├── pyproject.toml                   # Root project, uv build backend
├── uv.lock                          # Locked dependency versions
└── .python-version                  # Pinned interpreter (3.12)
```

Chapter *N* of the book maps to `theory/chapters/0N_*/` and `projects/0N_*/`.

## Conventions

- **Theory before code.** Each chapter's mathematics is derived by hand in
  `theory/` before the matching project is built in `projects/`.
- **One workflow per chapter:** write the theory, build the implementation, wire
  it to CI, and merge via pull request.
- **Python end to end.** Data work uses pandas or Polars, never SQL, and the
  capstone stores its state in JSONL and Parquet instead of a database. Keeping
  to one language keeps the focus on modelling and systems design.
- **Locked dependencies.** `uv` with a committed `uv.lock` and a pinned
  interpreter. There is no global `requirements.txt`. A project whose
  requirements conflict with the root gets its own `pyproject.toml` or
  `Dockerfile`.
- **Data stays out of Git.** Datasets are tracked with DVC and stored in remote
  storage.
- **`main` is protected.** Work happens on feature branches and merges through
  pull requests that pass their status checks.
- **Production-ready Python.** Type hints throughout, docstrings on public
  interfaces, and `pytest` coverage for library code.
- **Modular LaTeX.** Chapter prose lives in its own chapter file; `main.tex`
  only `\input`s the chapters.

## Chapters

### Phase 1 — Mathematical & Engineering Foundations

#### Chapter 1: Mathematical Foundations for Machine Learning

- [ ] **Linear Algebra** — vectors, matrices, tensors, eigendecomposition, SVD
- [ ] **Matrix Calculus** — gradients, chain rule, Jacobians, Hessians, Taylor series
- [ ] **Multivariate Optimization** — convexity, Lagrange multipliers, KKT, gradient descent
- [ ] **Numerical Computing & Stability** — floating point, cancellation, log-sum-exp, conditioning
- [ ] **Probability & Statistics** — distributions, Bayes' theorem, MLE, MAP, covariance
- [ ] **Statistical Inference & Experiment Design** — CLT, bootstrap, hypothesis tests, power, A/B testing
- [ ] **Information Theory** — entropy, cross-entropy, KL divergence

#### Chapter 2: Engineering Foundations & Infrastructure

- [ ] **Systems Design Principles** — separation of concerns, idempotency, reproducibility
- [ ] **Monorepo Architecture** — shared library vs per-project dependency boundaries
- [ ] **Isolated Environments** — virtual environments, lockfiles vs version ranges
- [ ] **Dependency Management with `uv`** — `uv sync` / `add` / `run`, `uv.lock`, pinned interpreter
- [ ] **Data Version Control** — DVC pointers, remote storage, `dvc pull` / `push`
- [ ] **Git Branch Protection** — feature branches, required reviews and checks, linear history
- [ ] **CI/CD with GitHub Actions** — path filters, PDF build, `pytest`, lint and format gates
- [ ] **Core Software Engineering** — `pytest` fixtures and parametrisation, type hints, docstrings

### Phase 2 — Applied Machine Learning & AI

#### Chapter 3: Data, Classical ML & Model Evaluation

- [ ] **Data Loading & EDA** — pandas/Polars, vectorisation, joins, `groupby`, profiling
- [ ] **Data Cleaning** — missingness mechanisms, imputation, outliers, dtypes
- [ ] **Feature Engineering** — scaling, categorical encoding, interactions, cyclical features
- [ ] **Dimensionality Reduction** — PCA, t-SNE, UMAP
- [ ] **Linear Models** — OLS, logistic regression, Ridge and Lasso
- [ ] **Tree-Based Models** — decision trees, random forests, gradient boosting (XGBoost, LightGBM)
- [ ] **Support Vector Machines** — primal and dual, kernel trick, max margin
- [ ] **Unsupervised Learning** — K-Means, hierarchical clustering, PCA derivation
- [ ] **Recommender Systems** — collaborative filtering, matrix factorisation, cold start, MAP@k, NDCG
- [ ] **Bias-Variance Tradeoff** — deriving the error decomposition
- [ ] **Validation Strategies** — k-fold, stratified, temporal splits
- [ ] **Metrics** — confusion matrix, precision/recall/F1, ROC-AUC, RMSE, MAE, R²
- [ ] **Hyperparameter Optimization** — grid/random search, successive halving, Bayesian optimisation, Optuna
- [ ] **Model Interpretability** — SHAP, permutation importance, PDPs, LIME
- [ ] **Data Issues** — class imbalance, SMOTE, leakage, fairness definitions
- [ ] **Experiment Tracking** — MLflow or Weights & Biases

#### Chapter 4: Bayesian ML & Probabilistic Modeling

- [ ] **Core Bayesian Inference** — the Bayes integral, conjugate priors
- [ ] **Markov Chain Monte Carlo** — detailed balance, Metropolis-Hastings, HMC
- [ ] **Variational Inference** — deriving the ELBO
- [ ] **Gaussian Processes** — kernels, predictive mean and variance
- [ ] **Bayesian Deep Learning** — Bayesian neural networks, MC Dropout

#### Chapter 5: Time Series Analysis & Financial Engineering

- [ ] **Classical Time Series** — stationarity, ACF/PACF, ADF test, ARIMA/SARIMA
- [ ] **Volatility Modeling** — ARCH, GARCH
- [ ] **Stochastic Calculus & Quant Finance** — Brownian motion, Itô's lemma, Black-Scholes, Markowitz
- [ ] **Deep Time Series** — TCNs, PatchTST
- [ ] **Backtesting** — walk-forward validation, look-ahead and survivorship bias

#### Chapter 6: Deep Learning Fundamentals

- [ ] **Multilayer Perceptron** — forward pass and backpropagation by hand
- [ ] **Activation Functions** — sigmoid, tanh, ReLU, GELU, softmax and their derivatives
- [ ] **Optimization Algorithms** — SGD, momentum, RMSProp, Adam
- [ ] **Loss Functions** — MSE, cross-entropy, hinge
- [ ] **Regularization** — dropout, batch normalisation, early stopping
- [ ] **Sequence Models** — RNNs, BPTT, LSTM/GRU, seq2seq, Bahdanau attention
- [ ] **Efficient Training & HPC** — GPU model, memory arithmetic, mixed precision, DDP/FSDP, SLURM

#### Chapter 7: Computer Vision & Scientific ML

- [ ] **CNNs** — convolution, pooling, stride, padding, receptive fields
- [ ] **Classic Architectures** — ResNet, VGG
- [ ] **Object Detection & Segmentation** — YOLO, Mask R-CNN, IoU
- [ ] **Modern CV** — Vision Transformers, patch embeddings
- [ ] **Physics-Informed Neural Networks** — PDE residuals in the loss
- [ ] **Neural Operators** — Fourier Neural Operator

#### Chapter 8: Generative AI, NLP & Agentic Systems

- [ ] **Foundational Generative Models** — VAEs, GANs, diffusion models
- [ ] **Transformer Architecture** — scaled dot-product attention, positional encoding
- [ ] **Parameter-Efficient Fine-Tuning** — LoRA
- [ ] **Local Inference & Quantization** — INT8/INT4, KV cache
- [ ] **Hugging Face Ecosystem** — `transformers`, `datasets`, the Hub
- [ ] **Reinforcement Learning Fundamentals** — MDPs, Bellman equations, Q-learning, policy gradients, PPO
- [ ] **Alignment** — SFT, reward modelling, RLHF, DPO
- [ ] **Prompt Engineering** — few-shot, chain-of-thought, tree-of-thought
- [ ] **Retrieval-Augmented Generation** — vector stores, similarity metrics, chunking, embeddings
- [ ] **Agentic Systems** — ReAct, tool calling, LangGraph
- [ ] **Generative Evaluation** — BLEU, ROUGE, perplexity, RAG metrics
- [ ] **Adversarial Robustness & AI Security** — FGSM/PGD, poisoning, prompt injection, privacy

### Phase 3 — Production & Deployment

#### Chapter 9: End-to-End Systems Design & MLOps (Capstone)

- [ ] **From Notebooks to Live Services** — training/serving boundary, model versioning, skew
- [ ] **Data Persistence for Services** — append-only logs, schema evolution, non-blocking writes
- [ ] **Microservice Architecture** — statelessness, health checks, twelve-factor config
- [ ] **RESTful APIs with FastAPI** — Pydantic schemas, dependency injection, `/predict`
- [ ] **Prediction Logging** — JSONL records, Parquet analysis, partitioning
- [ ] **Containerization & Orchestration** — multi-stage Docker builds, `docker compose`
- [ ] **Model Serving at Scale** — vLLM, llama.cpp, batching, latency vs throughput
- [ ] **Cloud Deployment** — AWS EC2, S3, SageMaker
- [ ] **Monitoring** — structured logging, drift detection, retraining triggers
