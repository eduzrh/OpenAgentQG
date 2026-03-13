"""OpenAgentQG configuration."""
import os

# API: read from env only (no keys in repo). Set: export OPENAI_API_KEY=sk-xxx [OPENAI_API_BASE=...]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE = (os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1") or "https://api.openai.com/v1").rstrip("/")

# LLM: GPT-4 ~34 BLEU-4/63 ROUGE-L; gpt-3.5-turbo ~10–12/35–39
LLM_MODEL = os.getenv("OPENAGENTQG_MODEL", "gpt-3.5-turbo-1106")

# Neuro-symbolic fusion
ENTROPY_THRESHOLD = 0.5  # tau: accept parametric knowledge if delta_H <= tau
META_NEURAL_TOP_K = 3   # max virtual nodes per subgraph

# Agentic generation
QUALITY_ACCEPT_THRESHOLD = 4   # tau_accept: accept QA if score >= 4
MAX_ITERATIONS = 3             # max MDP iterations for callbacks
MAX_WORKERS = 10               # parallel workers
BATCH_QS_PER_CALL = 20         # questions per API call (batch mode)

# Paper full vs fast: PAPER_FULL=1: Stage1 neuro-symbolic fusion + Stage2 full agentic MDP. Fast: triples only + one-shot.
PAPER_FULL_MODE = os.environ.get("OPENAGENTQG_PAPER_FULL", "0") == "1"
# Ablation (Table 5): no_fusion|no_agentic|no_meta_knowledge|no_graph_construction|no_core_role_mgmt|no_collaborative_decision|no_agentic_execution|no_quality_assessment
ABLATION_MODE = (os.environ.get("OPENAGENTQG_ABLATION", "") or "").strip().lower()
if PAPER_FULL_MODE:
    FAST_MODE = False
    ONE_SHOT_Q = False
else:
    FAST_MODE = True
    ONE_SHOT_Q = True
N_FEW_SHOT = 3
USE_SUBGRAPH_PROMPT = True

# Paths: data in data/ (mhqg-wq=WQ, mhqg-pq=PQ)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
# Graph2Seq: set GRAPH2SEQ_ROOT env to path of Graph2Seq4TKGQG (see README for link)
GRAPH2SEQ_ROOT = os.environ.get("GRAPH2SEQ_ROOT", "")
# Example bank and prompt tuning
EXAMPLE_BANK_PATH = os.path.join(DATA_ROOT, "mhqg-wq", "example_bank.json")
NUM_EXAMPLES_FROM_BANK = 20
USE_SIMILAR_EXAMPLES = True
DOUBLE_GENERATION = True   # generate 2, pick better by gold-style proxy
TRIPLE_GENERATION = True   # generate 3, pick best
NUM_TRAIN_FULL_IN_PROMPT = 18
NUM_TRAIN_STYLE_QUESTIONS = 22
USE_TOP1_SIMILAR_FIRST = True
NUM_TOP_SIMILAR_FIRST = 2  # top similar examples first
CLONE_CLOSEST_STRUCTURE = True  # clone structure from closest example
USE_DENSE_RETRIEVAL = True   # dense retrieval by triples+answer embedding
EMBEDDING_MODEL = "text-embedding-3-small"
USE_TWO_STAGE_GENERATION = True  # plan relation path then generate
FIVE_WAY_GENERATION = False
REVISE_QUESTION = False
