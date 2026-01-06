"""
LightRAG configuration constants and query modes.
"""

# Supported query modes in LightRAG
QUERY_MODES = {
    "naive",    # Simple keyword matching
    "local",    # Local context search
    "global",   # Global knowledge graph search
    "hybrid"    # Combination of local and global
}

# Default query mode
DEFAULT_MODE = "hybrid"

# Default working directory for LightRAG cache
DEFAULT_WORKING_DIR = "cache/lightrag"
