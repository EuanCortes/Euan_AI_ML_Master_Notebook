# The Applied AI Engineer's Compendium: Project Plan & Master Checklist

## Part 1: Master Project Plan

### 1. Project Overview
This repository serves as a living portfolio, combining rigorous mathematical theory (via LaTeX) with production-ready machine learning engineering (via containerized Python code). 

### 2. Repository Architecture
Maintain a modular structure to ensure the repository remains scalable as new chapters and projects are added.

```text
master-ai-portfolio/
├── theory_book/                 # LaTeX source code for the master notebook
│   ├── main.tex
│   ├── chapters/                # Modular chapter files (e.g., 01_math_foundations.tex)
│   └── figures/
├── projects/                    # Executable code linking to theory
│   ├── 02_classical_ml/         # Projects matching Chapter 2
│   │   └── housing_price_xgboost/
│   ├── 08_gen_ai/               # Projects matching Chapter 8
│   │   └── local_rag_pipeline/
├── docs/                        # Project planning and reference materials
│   ├── AI_ML_Master_Checklist.md
│   └── Project_Plan.md
└── README.md

### 3. Future-Proofing Guidelines
*   **Modular LaTeX:** Use `\input{chapters/...}` in `main.tex`. Avoid a monolithic text file.
*   **Automated PDF Compilation:** Implement GitHub Actions to compile the LaTeX code into a PDF automatically upon merging to the `main` branch.
*   **Isolated Dependencies:** Do not use a global `requirements.txt`. Give each project folder its own `pyproject.toml` (via Poetry or uv) or a dedicated `Dockerfile`.
*   **Data Version Control (DVC):** Do not commit large datasets to Git. Use DVC to create lightweight pointers to data stored in a cloud bucket (AWS S3, Google Drive), ensuring reproducibility.

### 4. Execution Strategy
*   **Phase 1:** Set up the Git repository, folder structure, and initial LaTeX boilerplate.
*   **Phase 2:** Work through the master checklist chapter by chapter.
*   **Phase 3:** Write the theoretical math and explanations in the physical notebook/LaTeX.
*   **Phase 4:** Build the corresponding Python implementation in the `projects/` directory.

---

## Part 2: The Master Checklist

### 1. Mathematical Foundations
*Before you can build the models, you need to speak the language they are written in.*

*   **Linear Algebra:** Vectors, matrices, tensors, dot products, eigenvalue/eigenvector decomposition, Singular Value Decomposition (SVD) ($A = U \Sigma V^T$).
*   **Calculus:** Partial derivatives, chain rule (the heart of backpropagation), gradients, Hessians, Taylor series expansion.
*   **Probability & Statistics:** 
    *   Probability distributions (Normal, Poisson, Bernoulli, Beta, Dirichlet).
    *   Bayes' Theorem, Maximum Likelihood Estimation (MLE), Maximum A Posteriori (MAP).
    *   Expected value, variance, covariance matrices.
*   **Information Theory:** Entropy ($H(X)$), Cross-Entropy, Kullback-Leibler (KL) Divergence ($D_{KL}(P||Q)$).

### 2. General Machine Learning (Classical ML)
*   **Linear Models:** Ordinary Least Squares (OLS) closed-form solution ($\beta = (X^TX)^{-1}X^TY$), Logistic Regression (Sigmoid function mapping), Ridge (L2) and Lasso (L1) regularization penalties.
*   **Tree-Based Models:** Decision Trees (Information Gain, Gini impurity equations), Random Forests (bagging math), Gradient Boosting Machines (XGBoost, LightGBM—derive the Taylor expansion of the loss function).
*   **Support Vector Machines (SVM):** The primal and dual optimization problems, the kernel trick (RBF, polynomial), max-margin classification.
*   **Unsupervised Learning:** K-Means clustering (Lloyd's algorithm), Hierarchical clustering, Principal Component Analysis (PCA) (deriving PCA via eigenvalue decomposition of the covariance matrix).

### 3. Bayesian Machine Learning & Probabilistic Modeling
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

### 4. Time Series Analysis & Financial Engineering
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

### 5. Classical Model Evaluation, Bias, & Lifecycle
*   **The Bias-Variance Tradeoff:** Deriving $Error = Bias^2 + Variance + Irreducible Error$.
*   **Validation Strategies:** K-fold cross-validation, stratified sampling, temporal splits (crucial for time series!).
*   **Metrics:** Confusion matrix, Precision, Recall, F1-Score, ROC-AUC, RMSE, MAE, R-squared.
*   **Data Issues:** Imbalanced datasets (SMOTE math), data leakage, algorithmic bias and fairness definitions (Demographic Parity, Equalized Odds).

### 6. Deep Learning Fundamentals
*   **The Multilayer Perceptron (MLP):** Nodes, weights, biases. Write out the forward pass and backpropagation (chain rule) by hand.
*   **Activation Functions:** Sigmoid, Tanh, ReLU, GELU, Softmax (write their equations and first derivatives).
*   **Optimization Algorithms:** Gradient Descent update rules, Momentum, RMSProp, Adam (derive the bias-corrected first and second moment estimates).
*   **Loss Functions:** MSE, Binary/Categorical Cross-Entropy, Hinge Loss.
*   **Regularization:** Dropout, Batch Normalization (write the mean/variance centering equations), Early Stopping.

### 7. Computer Vision (CV)
*   **CNNs:** Convolution operations (write the discrete convolution math), pooling layers, strides, padding, receptive fields.
*   **Classic Architectures:** ResNet (skip connection math: $H(x) = F(x) + x$), VGG.
*   **Object Detection & Segmentation:** YOLO, Mask R-CNN, Intersection over Union (IoU) calculation.
*   **Modern CV:** Vision Transformers (ViT) and image patch embedding.

### 8. Generative AI & Natural Language Processing
*   **Foundational Generative Models:** 
    *   VAEs: The reparameterization trick ($z = \mu + \sigma \odot \epsilon$).
    *   GANs: The minimax game loss function.
    *   Diffusion Models (DDPMs): The Markov chain forward noise equations and reverse denoising U-Net process.
*   **Large Language Models (LLMs):** 
    *   The Transformer Architecture: Write out the exact Scaled Dot-Product Attention equation ($Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$).
    *   Positional Encoding (sine/cosine equations).
*   **Parameter-Efficient Fine-Tuning (PEFT):** LoRA math (low-rank matrix decomposition $W = W_0 + BA$).
*   **Alignment:** RLHF (Reward modeling, PPO math), DPO (Direct Preference Optimization loss).
*   **Local Inference & Quantization:** Math behind compression (FP16 to INT8/INT4), KV Cache mechanics.

### 9. AI & Agentic Deployment
*   **Prompt Engineering:** Few-shot, Chain-of-Thought (CoT), Tree-of-Thought.
*   **Retrieval-Augmented Generation (RAG):**
    *   Vector databases (Chroma, FAISS).
    *   Distance metrics: Cosine similarity ($\frac{A \cdot B}{||A|| ||B||}$), Euclidean distance.
*   **Agentic Systems:** ReAct framework, tool/function calling logic, Directed Acyclic Graphs (DAGs) for multi-agent orchestration (LangGraph).
*   **Generative Evaluation & Benchmarking:**
    *   *N-gram metrics:* BLEU, ROUGE formulas.
    *   *Statistical metrics:* Perplexity ($2^{H(P)}$), Cross-Entropy.
    *   *RAG metrics:* Context precision/recall, faithfulness.

### 10. Scientific Machine Learning (SciML)
*   **Physics-Informed Neural Networks (PINNs):** Embedding partial differential equations (PDEs) directly into the neural network loss function ($Loss = Loss_{data} + Loss_{PDE}$).
*   **Neural Operators:** Fourier Neural Operator (FNO) and solving PDEs in the frequency domain.

### 11. MLOps, Engineering & Software Architecture
*   **Core Software Engineering:** Unit testing (`pytest`), RESTful API principles.
*   **Version Control & CI/CD:** Git workflows, GitHub Actions (automated testing and deployment).
*   **Experiment Tracking:** Weights & Biases (W&B), MLflow.
*   **Hugging Face Ecosystem:** `transformers`, `datasets`, Hub management.
*   **Cloud & Containerization:** Docker architecture, AWS (EC2, S3, SageMaker).
*   **Model Serving:** FastAPI, Inference Engines (`vLLM`, `llama.cpp`).