"""
DAT Strategies and Prompts Configuration.
Extracted from DAT_GPT API scripts.
"""

# DAT Task Prompts
DAT_STRATEGIES = {
    "none": (
        "Please enter 10 words that are as different from each other as possible, "
        "in all meanings and uses of the words. "
        "Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). "
        "No proper nouns (e.g., no specific people or places). "
        "No specialised vocabulary (e.g., no technical terms). "
        "Think of the words on your own (e.g., do not just look at objects in your surroundings). "
        "Make a list of these 10 words, a single word in each entry of the list."
    ),
    
    "thesaurus": (
        "Please enter 10 words that are as different from each other as possible, "
        "in all meanings and uses of the words, using a strategy that relies on using a thesaurus. "
        "Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). "
        "No proper nouns (e.g., no specific people or places). "
        "No specialised vocabulary (e.g., no technical terms). "
        "Think of the words on your own (e.g., do not just look at objects in your surroundings). "
        "Make a list of these 10 words, a single word in each entry of the list."
    ),
    
    "etymology": (
        "Please enter 10 words that are as different from each other as possible, "
        "in all meanings and uses of the words, using a strategy that relies on varying etymology. "
        "Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). "
        "No proper nouns (e.g., no specific people or places). "
        "No specialised vocabulary (e.g., no technical terms). "
        "Think of the words on your own (e.g., do not just look at objects in your surroundings). "
        "Make a list of these 10 words, a single word in each entry of the list."
    ),
    
    "opposites": (
        "Please enter 10 words that are as different from each other as possible, "
        "in all meanings and uses of the words, using a strategy that relies on meaning opposition. "
        "Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). "
        "No proper nouns (e.g., no specific people or places). "
        "No specialised vocabulary (e.g., no technical terms). "
        "Think of the words on your own (e.g., do not just look at objects in your surroundings). "
        "Make a list of these 10 words, a single word in each entry of the list."
    ),
    
    "random": (
        "Please enter 10 words that are as different from each other as possible, "
        "in all meanings and uses of the words, using a strategy that relies on randomness. "
        "Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). "
        "No proper nouns (e.g., no specific people or places). "
        "No specialised vocabulary (e.g., no technical terms). "
        "Think of the words on your own (e.g., do not just look at objects in your surroundings). "
        "Make a list of these 10 words, a single word in each entry of the list."
    ),
}

# Creative Writing Task Prompts (from other scripts)
CREATIVE_PROMPTS = {
    "synopsis": "Write a creative and unique movie synopsis that is about 100 words long.",
    "flash_fiction": "Write a creative flash fiction story in exactly 100 words.",
    "haiku": "Write a creative haiku following the 5-7-5 syllable pattern."
}

# Default temperature settings for different strategies
DEFAULT_TEMPERATURES = {
    "none": 0.7,
    "thesaurus": 0.7,
    "etymology": 0.7,
    "opposites": 0.7,
    "random": 1.0,  # Higher temperature for random strategy
}

# Model-specific configurations
MODEL_CONFIGS = {
    "openai": {
        "max_tokens": 500,
        "default_model": "gpt-5.1-mini",
    },
    "anthropic": {
        "max_tokens": 500,
        "default_model": "claude-3-5-sonnet-latest",
    },
    "gemini": {
        "max_tokens": 500,
        "default_model": "gemini-1.5-flash-latest",
    },
    "ollama": {
        "max_tokens": 500,
        "default_model": "llama3.2",
    }
}