#!/usr/bin/env bash
# Builds eval/.ragas_venv -- an isolated environment for RAGAS scoring,
# pinned via requirements-ragas.txt. Isolated deliberately: ragas's
# langchain-core/-community/-openai pins conflict with the rest of this
# project's main venv (found the hard way -- installing ragas there broke
# langgraph/langchain-openai; do not repeat that, always use this venv).
#
# Usage: bash eval/setup_ragas_env.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

python3 -m venv .ragas_venv
./.ragas_venv/bin/pip install -q --upgrade pip
./.ragas_venv/bin/pip install -q -r requirements-ragas.txt

./.ragas_venv/bin/python -c "
import ragas
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
print(f'ragas {ragas.__version__} -- environment OK')
"
