"""
core/llm/registry.py — LLM provider registry.

Add new providers here. Each entry describes how to connect and which
prompt style to use. The 'chat_style' key maps to the backend used by
LLMCaller: 'kobold' | 'openai' | 'mistral_conv'.
"""

PROVIDER_REGISTRY: dict[str, dict] = {
    "koboldcpp": {
        "label":          "KoboldCPP (local)",
        "base_url":       "http://localhost:5001",
        "needs_api_key":  False,
        "needs_agent_id": False,
        "needs_model":    False,
        "chat_style":     "kobold",
    },
    "mistral": {
        "label":          "Mistral API",
        "base_url":       "https://api.mistral.ai",
        "needs_api_key":  True,
        "needs_agent_id": True,
        "needs_model":    False,
        "chat_style":     "mistral_conv",
    },
    # "lmstudio": {
    #     "label":          "LM Studio (local)",
    #     "base_url":       "http://localhost:1234",
    #     "needs_api_key":  False,
    #     "needs_agent_id": False,
    #     "needs_model":    True,
    #     "chat_style":     "openai",
    # },
    # "ollama": {
    #     "label":          "Ollama (local)",
    #     "base_url":       "http://localhost:11434",
    #     "needs_api_key":  False,
    #     "needs_agent_id": False,
    #     "needs_model":    True,
    #     "chat_style":     "openai",
    # },
    # "llamacpp": {
    #     "label":          "llama.cpp server (local)",
    #     "base_url":       "http://localhost:8080",
    #     "needs_api_key":  False,
    #     "needs_agent_id": False,
    #     "needs_model":    False,
    #     "chat_style":     "openai",
    # },
}
