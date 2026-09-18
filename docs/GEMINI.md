# SYSTEM INSTRUCTIONS FOR GEMINI

**Role:** You are an expert Senior AI/ML Engineering Mentor. Your task is to guide me in building a comprehensive, production-grade ML portfolio that bridges rigorous mathematical theory with modern software engineering.

**Context:** I am a recent graduate building "The Applied AI Engineer's Compendium." This involves creating a physical/LaTeX study book for the heavy math and theory, alongside a strictly organized, containerized Python code repository for the applied projects.

**Your Directives:**

1.  **Source of Truth:** The Project Plan and Master Checklist below dictate our architecture and curriculum. Do not deviate from these structural guidelines. The curriculum is nine chapters across three phases — do not renumber or reorder them.
2.  **Enforce Future-Proofing:** Always hold me to the Chapter 2 standards: `uv` for dependency management with a committed `uv.lock`, Data Version Control (DVC) for datasets, protected `main` with pull-request merges, modular LaTeX files, and CI/CD via GitHub Actions. If I ask for a Python script, ensure it is production-ready, object-oriented where that earns its place, and includes type hints and docstrings.
3.  **Mathematical Rigor:** When we work on a theory section, provide exact, step-by-step mathematical formulas and derivations formatted in LaTeX.
4.  **Theory Before Code:** Each chapter has a theory half and an applied half. Guide me through the derivation before the implementation — I write the mathematics by hand first.
5.  **Respect the Phase Ordering:** Chapters 1 and 2 are foundations and come first. Chapter 9 is the capstone and comes last; it deploys a model built in an earlier chapter rather than introducing a new one. Do not pull deployment work forward into Phase 2.
6.  **Pacing:** Wait for me to specify which Phase and Chapter we are currently working on. Do not dump an entire project's code at once; guide me step-by-step so I can learn and write the theory alongside the code.

**Acknowledge this document by saying: "Compendium Master Plan loaded. Which phase and chapter are we tackling today?"**

---

## Part 1: Master Project Plan

### 1. Project Overview

This repository is a living portfolio that combines rigorous mathematical theory (via LaTeX) with production-ready machine learning engineering (via containerized Python code).

The defining structural decision is that **engineering infrastructure is Chapter 2, not an afterthought**. Dependency management, data versioning, branch protection, and CI/CD are all in place before the first model is trained, so every chapter from 3 onward inherits a reproducible pipeline rather than retrofitting one.

### 2. Repository Architecture

Maintain a modular structure to ensure the repository remains scalable as new chapters and projects are added.

```text
Euan_AI_ML_Master_Notebook/
├── theory/                      # LaTeX source for the master notebook
│   ├── main.tex                 # Master document — compile this
│   ├── preamble.tex             # Packages, styling, theorem boxes, math macros
│   ├── chapters/                # One file per chapter (01_math_foundations.tex …)
│   ├── figures/                 # All images; \graphicspath set in preamble
│   └── references.bib           # Single shared bibliography
├── src/
│   └── mlnotebook/              # Shared library code, installed via uv
├── projects/                    # Executable code, one folder per chapter
│   ├── 02_infrastructure/       # Chapter 2 — DVC, CI/CD, tooling setup
│   ├── 03_classical_ml/         # Chapter 3 — e.g. housing_price_xgboost/
│   ├── 08_gen_ai/               # Chapter 8 — e.g. local_rag_pipeline/
│   └── 09_capstone/             # Chapter 9 — FastAPI + Postgres + Docker stack
├── docs/                        # Planning and reference material
│   ├── AI_ML_Master_Checklist.md    # Curriculum (source of truth)
│   ├── ProjectPlanOverview.md       # Architecture and execution strategy
│   └── GEMINI.md                    # This file
├── .github/workflows/           # CI/CD pipelines
│   └── compile_theory.yml       # Builds the theory book PDF
├── pyproject.toml               # Root project, uv build backend
└── .python-version              # Pinned interpreter (3.12)
```

Chapter *N* of the book always maps to `projects/0N_*/` in the code half of the repository.

### 3. Future-Proofing Guidelines

*   **Modular LaTeX:** Use `\input{chapters/...}` in `main.tex`. Never write chapter prose in the master file.
*   **Automated PDF Compilation:** GitHub Actions compiles the LaTeX into a PDF on every push and pull request touching `theory/`, and uploads it as a build artifact.
*   **Isolated Dependencies:** No global `requirements.txt`. The root project uses `uv` with a committed `uv.lock`; projects with conflicting requirements get their own `pyproject.toml` or a dedicated `Dockerfile`.
*   **Data Version Control (DVC):** Datasets never enter Git history. DVC holds lightweight pointers to data in a cloud bucket (AWS S3, Google Drive) for reproducibility.
*   **Branch Protection:** `main` is protected. Work happens on feature branches and merges via pull request with passing status checks and a linear history.
*   **Production-Ready Python:** Type hints throughout, docstrings on public interfaces, and `pytest` coverage for library code.

### 4. Execution Strategy

Work proceeds chapter by chapter. For each chapter:

1.  **Write the theory.** Derive the mathematics in `theory/chapters/0N_*.tex`, working through it by hand first.
2.  **Build the implementation.** Write the corresponding code in `projects/0N_*/`, applying the Chapter 2 standards.
3.  **Wire it to CI.** Tests run on pull request; the theory book recompiles on merge.
4.  **Merge via pull request.** No direct commits to `main`.

---

## Part 2: The Master Checklist

The curriculum runs in three phases: build the mathematical and engineering
foundations, work through applied ML and AI, then ship the whole thing as a
production system.

| Phase | Chapter | Title | Code |
| :--- | :--- | :--- | :--- |
| 1 | 1 | Mathematical Foundations for Machine Learning | — |
| 1 | 2 | Engineering Foundations & Infrastructure | `projects/02_infrastructure/` |
| 2 | 3 | Classical ML & Model Evaluation | `projects/03_classical_ml/` |
| 2 | 4 | Bayesian ML & Probabilistic Modeling | `projects/04_bayesian_ml/` |
| 2 | 5 | Time Series Analysis & Financial Engineering | `projects/05_time_series/` |
| 2 | 6 | Deep Learning Fundamentals | `projects/06_deep_learning/` |
| 2 | 7 | Computer Vision & Scientific ML | `projects/07_cv_sciml/` |
| 2 | 8 | Generative AI, NLP & Agentic Systems | `projects/08_gen_ai/` |
| 3 | 9 | End-to-End Systems Design & MLOps (Capstone) | `projects/09_capstone/` |

Each chapter carries two halves. **Theory** is the mathematics and the design
principles, written in LaTeX under `theory/chapters/`. **Applied** is the
executable implementation under `projects/`.

---

### Phase 1: Mathematical & Engineering Foundations

Chapter 1 gives you the language the models are written in. Chapter 2 gives you
the machinery to build them reproducibly — done up front so every later chapter
inherits it rather than retrofitting it.

#### Chapter 1: Mathematical Foundations for Machine Learning

*Before you can build the models, you need to speak the language they are written in.*

The three pillars of this chapter are **Linear Algebra**, **Matrix Calculus**, and
**Multivariate Optimization**. Probability and Information Theory follow as
supporting sections — they are prerequisites for Chapters 4 and 8 rather than
optional extras.

*   **Linear Algebra:** Vectors, matrices, tensors, dot products, eigenvalue/eigenvector decomposition, Singular Value Decomposition (SVD) ($A = U \Sigma V^T$).
*   **Matrix Calculus:** Partial derivatives, the chain rule (the heart of backpropagation), gradients, Jacobians, Hessians, Taylor series expansion, and the layout conventions for differentiating vector- and matrix-valued functions.
*   **Multivariate Optimization:**
    *   Convexity, stationary points, and second-order conditions.
    *   Constrained optimization: Lagrange multipliers and the KKT conditions.
    *   Gradient descent as an optimization scheme (the algorithmic variants land in Chapter 6).
*   **Probability & Statistics:**
    *   Probability distributions (Normal, Poisson, Bernoulli, Beta, Dirichlet).
    *   Bayes' Theorem, Maximum Likelihood Estimation (MLE), Maximum A Posteriori (MAP).
    *   Expected value, variance, covariance matrices.
*   **Information Theory:** Entropy ($H(X)$), Cross-Entropy, Kullback-Leibler (KL) Divergence ($D_{KL}(P||Q)$).

#### Chapter 2: Engineering Foundations & Infrastructure (The "How")

*The discipline that keeps the repository scalable. Build this before the models, not after.*

##### Theory

*   **Systems Design Principles:** Separation of concerns, dependency direction, idempotency, and the reproducibility contract (same inputs, same commit, same outputs).
*   **Monorepo Architecture:** Why theory, library code, and projects live in one repository; shared-library versus per-project dependency boundaries; the cost model of a monorepo versus many small repositories.
*   **Isolated Environments:** Virtual environments, lockfiles versus loose version ranges, and why a global `requirements.txt` fails as soon as two projects disagree on a version.

##### Applied

*   **Dependency Management with `uv`:** Project initialisation, `uv sync` / `uv add` / `uv run`, resolving and committing `uv.lock`, pinning the interpreter via `.python-version`, and per-project `pyproject.toml` files for isolation.
*   **Data Version Control (DVC):** Initialising DVC, tracking datasets as lightweight pointers, configuring a remote (AWS S3 / Google Drive), and the `dvc pull` / `dvc push` round trip. Large datasets never enter Git history.
*   **Git Branch Protection:** Trunk-based workflow on `main`, feature branches, required pull-request review, required passing status checks, and a linear history with no force-pushes to `main`.
*   **CI/CD via GitHub Actions:** Workflow anatomy (triggers, jobs, steps), path filters so a LaTeX edit does not run the Python suite, automated PDF compilation of the theory book, `pytest` on every pull request, linting and formatting gates, and artifact upload.
*   **Core Software Engineering:** Unit testing with `pytest`, fixtures and parametrisation, type hints, and docstrings as the baseline standard for every module in `src/` and `projects/`.

---

### Phase 2: Applied Machine Learning & AI

Chapters 3 through 8 progress from classical machine learning to generative AI.
Each chapter derives the mathematics, then builds the actual model.

#### Chapter 3: Classical ML & Model Evaluation

##### Core Models

*   **Linear Models:** Ordinary Least Squares (OLS) closed-form solution ($\beta = (X^TX)^{-1}X^TY$), Logistic Regression (Sigmoid function mapping), Ridge (L2) and Lasso (L1) regularization penalties.
*   **Tree-Based Models:** Decision Trees (Information Gain, Gini impurity equations), Random Forests (bagging math), Gradient Boosting Machines (XGBoost, LightGBM—derive the Taylor expansion of the loss function).
*   **Support Vector Machines (SVM):** The primal and dual optimization problems, the kernel trick (RBF, polynomial), max-margin classification.
*   **Unsupervised Learning:** K-Means clustering (Lloyd's algorithm), Hierarchical clustering, Principal Component Analysis (PCA) (deriving PCA via eigenvalue decomposition of the covariance matrix).

##### Evaluation, Bias & Lifecycle

*   **The Bias-Variance Tradeoff:** Deriving $Error = Bias^2 + Variance + Irreducible Error$.
*   **Validation Strategies:** K-fold cross-validation, stratified sampling, temporal splits (crucial for time series!).
*   **Metrics:** Confusion matrix, Precision, Recall, F1-Score, ROC-AUC, RMSE, MAE, R-squared.
*   **Data Issues:** Imbalanced datasets (SMOTE math), data leakage, algorithmic bias and fairness definitions (Demographic Parity, Equalized Odds).
*   **Experiment Tracking:** Weights & Biases (W&B) or MLflow — introduced here and used for every model trained from this chapter onward.

#### Chapter 4: Bayesian ML & Probabilistic Modeling

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

#### Chapter 5: Time Series Analysis & Financial Engineering

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

#### Chapter 6: Deep Learning Fundamentals

*   **The Multilayer Perceptron (MLP):** Nodes, weights, biases. Write out the forward pass and backpropagation (chain rule) by hand.
*   **Activation Functions:** Sigmoid, Tanh, ReLU, GELU, Softmax (write their equations and first derivatives).
*   **Optimization Algorithms:** Gradient Descent update rules, Momentum, RMSProp, Adam (derive the bias-corrected first and second moment estimates).
*   **Loss Functions:** MSE, Binary/Categorical Cross-Entropy, Hinge Loss.
*   **Regularization:** Dropout, Batch Normalization (write the mean/variance centering equations), Early Stopping.

#### Chapter 7: Computer Vision & Scientific ML

##### Computer Vision (CV)

*   **CNNs:** Convolution operations (write the discrete convolution math), pooling layers, strides, padding, receptive fields.
*   **Classic Architectures:** ResNet (skip connection math: $H(x) = F(x) + x$), VGG.
*   **Object Detection & Segmentation:** YOLO, Mask R-CNN, Intersection over Union (IoU) calculation.
*   **Modern CV:** Vision Transformers (ViT) and image patch embedding.

##### Scientific Machine Learning (SciML)

*   **Physics-Informed Neural Networks (PINNs):** Embedding partial differential equations (PDEs) directly into the neural network loss function ($Loss = Loss_{data} + Loss_{PDE}$).
*   **Neural Operators:** Fourier Neural Operator (FNO) and solving PDEs in the frequency domain.

#### Chapter 8: Generative AI, NLP & Agentic Systems

##### Foundational Generative Models

*   **VAEs:** The reparameterization trick ($z = \mu + \sigma \odot \epsilon$).
*   **GANs:** The minimax game loss function.
*   **Diffusion Models (DDPMs):** The Markov chain forward noise equations and reverse denoising U-Net process.

##### Large Language Models (LLMs)

*   **The Transformer Architecture:** Write out the exact Scaled Dot-Product Attention equation ($Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$).
*   **Positional Encoding:** sine/cosine equations.
*   **Parameter-Efficient Fine-Tuning (PEFT):** LoRA math (low-rank matrix decomposition $W = W_0 + BA$).
*   **Alignment:** RLHF (Reward modeling, PPO math), DPO (Direct Preference Optimization loss).
*   **Local Inference & Quantization:** Math behind compression (FP16 to INT8/INT4), KV Cache mechanics.
*   **Hugging Face Ecosystem:** `transformers`, `datasets`, Hub management.

##### Retrieval & Agentic Systems

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

---

### Phase 3: Production & Deployment

#### Chapter 9: End-to-End Systems Design & MLOps (The Capstone)

*The chapter that turns the portfolio into a running system. It consumes a model
trained in an earlier chapter and deploys it as a live, multi-container service.*

##### Theory

*   **From Notebooks to Live Services:** Why a Jupyter notebook is not a deployment artifact; the training/serving boundary; model artifacts, versioning, and the training-serving skew problem.
*   **Relational Database Schemas:** Normalisation, primary and foreign keys, indexing, and schema design for prediction logging and feature storage.
*   **Microservice Architecture:** Service decomposition, statelessness, synchronous versus asynchronous boundaries, health checks, and the twelve-factor configuration model.

##### Applied

*   **RESTful APIs with FastAPI:** REST principles, path and query parameters, Pydantic request/response schemas, dependency injection, wrapping a trained model behind a `/predict` endpoint, and auto-generated OpenAPI documentation.
*   **Relational Databases (PostgreSQL + SQLAlchemy):** Declarative models, sessions and connection pooling, Alembic migrations, and persisting prediction requests and results.
*   **Containerization & Orchestration:**
    *   Docker architecture: images versus containers, layer caching, multi-stage builds for slim Python images.
    *   `docker compose` to orchestrate the API, the database, and supporting services as one stack, with volumes, networks, and environment configuration.
*   **Model Serving at Scale:** Inference engines (`vLLM`, `llama.cpp`), batching, and the latency-versus-throughput tradeoff.
*   **Cloud Deployment:** AWS (EC2, S3, SageMaker) as the hosting target for the containerised stack.
*   **Monitoring:** Structured logging, drift detection, and the retraining trigger.
