import os

# Keep automated tests independent from local .env model settings. Developer
# machines may enable large GPU models, but the test suite should stay fast and
# deterministic.
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["EMBEDDING_MODEL"] = "hash-local"
os.environ["EMBEDDING_DIMENSION"] = "128"
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["EMBEDDING_USE_FP16"] = "false"
os.environ["VECTOR_STORE_PROVIDER"] = "memory"
os.environ["RERANKER_PROVIDER"] = "none"
os.environ["LLM_PROVIDER"] = "template"
os.environ["LLM_MODEL"] = "template-local"
