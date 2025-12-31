# Syllabus — Self-Directed Computer Engineering & Machine Learning Program

## Program Intent

This syllabus defines a self-directed, project-based curriculum covering Computer Engineering, Machine Learning, and Intelligent Systems.

The program is designed to:

- Replace passive coursework with active implementation
- Emphasize real-world constraints and tradeoffs
- Produce public, verifiable engineering artifacts
- Align learning outcomes with real job requirements

Progress is measured by **delivered projects**, not time spent or content consumed.

---

## Program Structure

The program is divided into the following tracks:

0. Foundations (Math + Python)
1. Machine Learning
2. Signal Processing
3. Embedded Systems
4. Applied Natural Language Processing
5. Computer Vision
6. Distributed Systems
7. Optional: FPGA & Hardware Acceleration
8. Capstone Project

Tracks are sequential where dependencies exist, but some overlap is expected.

---

## Track 0 — Foundations (Math + Python)

### Objectives

- Translate mathematical concepts into working code
- Build intuition for optimization and learning algorithms
- Eliminate reliance on ML “black boxes”

### Core Topics

- Linear algebra: vectors, matrices, dot products, norms
- Probability: distributions, expectation, variance
- Statistics: sampling, noise, basic inference
- Optimization: gradient descent, loss functions
- Python: NumPy, basic data structures, plotting

### Required Projects

1. Linear regression from scratch
2. Logistic regression from scratch
3. Gradient descent visualization and analysis

### Exit Criteria

- Implement regression models without external references
- Explain gradient descent intuitively and mathematically
- Demonstrate understanding of loss surfaces

---

## Track 1 — Machine Learning

### Objectives

- Build, evaluate, and diagnose ML models
- Understand model bias, variance, and failure modes
- Design reproducible ML pipelines

### Core Topics

- Data cleaning and preprocessing
- Supervised learning
- Decision trees and ensemble methods
- Support Vector Machines and kernels
- Neural network fundamentals
- Unsupervised learning (k-means, PCA)

### Required Projects

1. Mini machine learning library (from scratch components)
2. End-to-end ML pipeline on a real dataset
3. Overfitting and bias analysis report

### Exit Criteria

- Justify model selection decisions
- Diagnose poor model performance
- Explain evaluation metrics and tradeoffs

---

## Track 2 — Signal Processing

### Objectives

- Analyze and transform real-world signals
- Improve ML performance through preprocessing
- Understand data in time and frequency domains

### Core Topics

- Sampling theory
- Discrete Fourier Transform (DFT / FFT)
- Filtering and denoising
- Noise modeling
- Feature extraction
- Stochastic processes

### Required Projects

1. Signal analysis toolkit (time + frequency domain)
2. Noise-robust classifier
3. Feature extraction comparison study

### Exit Criteria

- Interpret FFT results correctly
- Design and apply filters intentionally
- Demonstrate signal-aware ML improvements

---

## Track 3 — Embedded Systems

### Objectives

- Design and program sensor-based systems
- Work within real-time and resource constraints
- Bridge software, hardware, and data processing

### Core Topics

- C programming for microcontrollers
- GPIO, I2C, SPI, UART
- Sensor interfacing and drivers
- Timing and latency constraints
- Debugging embedded systems

### Required Projects

1. Custom sensor driver implementation
2. Real-time data acquisition and streaming system
3. Edge-based ML inference pipeline

### Exit Criteria

- Read and apply datasheets confidently
- Explain latency, memory, and power tradeoffs
- Debug hardware–software integration issues

---

## Track 4 — Applied Natural Language Processing

### Objectives

- Process and analyze unstructured text data
- Understand limitations of language models
- Apply NLP techniques to real domains

### Core Topics

- Tokenization and text normalization
- Bag-of-words and TF-IDF
- Word embeddings
- Sequence models
- Transformer concepts
- Generative model principles

### Required Projects

1. Keyword extraction system
2. Domain-specific NLP pipeline
3. Model comparison and error analysis

### Exit Criteria

- Explain representation choices
- Identify hallucination and bias risks
- Evaluate NLP systems beyond accuracy

---

## Track 5 — Computer Vision

### Objectives

- Extract meaning from images and video
- Combine classical vision and deep learning
- Evaluate performance under constraints

### Core Topics

- Image filtering and transformations
- Feature detection
- Convolutional Neural Networks (CNNs)
- Transfer learning
- Vision evaluation metrics

### Required Projects

1. End-to-end vision processing pipeline
2. Edge or constrained vision deployment
3. Accuracy vs latency tradeoff analysis

### Exit Criteria

- Explain preprocessing effects
- Justify architecture choices
- Evaluate robustness and performance

---

## Track 6 — Distributed Systems for ML

### Objectives

- Design scalable ML systems
- Understand platform tradeoffs
- Handle distributed execution and failure

### Core Topics

- Distributed workloads
- Model serving
- Edge vs cloud architectures
- Fault tolerance
- Data aggregation

### Required Projects

1. Distributed inference system
2. Sensor fleet aggregation backend
3. Failure-mode simulation and recovery study

### Exit Criteria

- Explain system architecture decisions
- Handle partial failure scenarios
- Measure and optimize performance

---

## Track 7 — Optional: FPGA & Hardware Acceleration

### Objectives

- Understand hardware acceleration for ML
- Explore performance optimization beyond CPUs

### Core Topics

- FPGA fundamentals
- ML inference acceleration concepts
- Performance benchmarking

### Required Project

1. Neural network acceleration prototype

### Exit Criteria

- Explain when acceleration is justified
- Compare latency and throughput tradeoffs

---

## Track 8 — Capstone Project

### Capstone Theme

**Smart Sensor System**

### Required Components

- Embedded sensing hardware
- Signal processing pipeline
- ML inference (edge and/or cloud)
- NLP and/or computer vision component
- Distributed data aggregation
- Performance evaluation and optimization

### Deliverables

- System architecture diagram
- Fully reproducible codebase
- Technical documentation
- Demo (video or live)
- Final written analysis

The capstone replaces a traditional thesis or degree credential.

---

## Completion Standard

This program is considered complete only when:

- All required projects are implemented
- Documentation is complete and clear
- Artifacts demonstrate real engineering competence
- The capstone meets all stated requirements

Time spent is irrelevant. Output is everything.
