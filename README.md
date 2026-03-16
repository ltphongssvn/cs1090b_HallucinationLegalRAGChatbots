# HallucinationLegalRAGChatbots

[![CI](https://github.com/ltphongssvn/cs1090b_HallucinationLegalRAGChatbots/actions/workflows/ci.yml/badge.svg)](https://github.com/ltphongssvn/cs1090b_HallucinationLegalRAGChatbots/actions/workflows/ci.yml)

GPU pipeline for comparing retrieval architectures (TF-IDF, CNN, LSTM, BERT bi-encoder, KG-augmented)
to reduce hallucination in legal RAG chatbots.

## Quick start
```bash
bash setup.sh                     # full GPU setup
SKIP_GPU=1 bash setup.sh          # CPU-only
DRY_RUN=1 bash setup.sh           # preview side effects
LOG_LEVEL=0 bash setup.sh         # quiet (CI)
```

## CI

GitHub Actions runs on every push/PR: lint → shell tests → unit tests → CPU smoke → security audit.
GPU tests run on the cluster only (`SKIP_GPU=1` in CI).
