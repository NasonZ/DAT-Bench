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
    
    "competitive": """You are a creative writer tasked with a challenge: to create a list of 10 words that are as different from each other as possible. You will be competing with 500 other writers to win 10.000 dollars. 

<tips>
Here are some tips from the organizer: 
1. Think of words that would never appear on the same page in a book, or even better: words that would never appear in the same book at all.
2. Think of words that are used in different contexts
3. Think of words that descibe different 'types'; physical things, social constructs, concepts, etc.
4. Before generating a new word always check ALL the previous words to see which contexts and types you already touched and therefore should be avoided
5. As you need to generate 10 words that are AS FAR from each other as possible, make sure you search for edge case words. Avoid too common/average words. 
</tips>

<rules>
Rules: 
1. Only single words in English. Only nouns (e.g., things, objects, concepts). 
2. No proper nouns (e.g., no specific people or places).
3. No specialised vocabulary (e.g., no technical terms).
4. Think of the words on your own (e.g., do not just look at objects in your surroundings).
5. Make a list of these 10 words, a single word in each entry of the list.
6. Output only the list, nothing else
</rules>"""
    ,
    
    "DAT_instructions": """You will be acting as a linguistic researcher. Your goal is to complete the divergent association task. 
    
Here is guidance which provides informaion and context on the task you need to complete:
<guidance>
About the task
The Divergent Association Task is a quick measure of verbal creativity and divergent thinking, the ability to generate diverse solutions to open-ended problems. The task involves thinking of 10 words that are as different from each other as possible. For example, the words cat and dog are similar, but the words cat and book are not. People who are more creative tend to generate words that have greater distances between them. These distances are inferred by examining how often the words are used together in similar contexts. Still, this task measures only a sliver of the complex process of creativity. See the frequently asked questions for more details.

We have validated this task on around 9,000 participants from 98 countries across the world. People who score higher on the task tend to be able to:

- think of novel and more varied uses for common objects (Alternative Uses Task)
- find associations between related words (e.g., giraffe and scarf; Bridge-the-Associative-Gap Task)
- solve more insight and analytical problems
</guidance>

<rules>
Rules
1. Only single words in English.
2. Only nouns (e.g., things, objects, concepts).
3. No proper nouns (e.g., no specific people or places).
4. No specialised vocabulary (e.g., no technical terms).
</rules>

Your goal is to produce a list of 10 words that are as different from each other as possible, in all meanings and uses of the words.""",
    
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

# Placeholder Query Decomposition Prompts (QD-DP) - Complex queries requiring orthogonal decomposition
QUERY_DECOMPOSITION_PROMPTS = {
    "housing_crisis": "How should the UK address its housing affordability crisis while balancing economic growth, environmental sustainability, and social equity?",
    
    "tech_monopolies": "How should regulators address the market dominance of big tech companies while preserving innovation, consumer benefits, and global competitiveness?",
    
    "aging_society": "How can societies adapt their economic models, healthcare systems, and urban planning to support rapidly aging populations?",
    
    "digital_privacy": "How can individuals maintain privacy and data sovereignty in an increasingly connected world of IoT devices, AI assistants, and digital services?",
}

# Task Decomposition Prompts (TD-DP) - Complex tasks requiring diverse approach generation
TASK_DECOMPOSITION_PROMPTS = {
    "rag_mvp": "Design and implement a production-ready RAG (Retrieval-Augmented Generation) system for a legal firm that needs to query case law, statutes, and internal documents with sub-10 second response times.",
    
    "documentation_tool": "Build an llm driven code documentation tool which uses doc strings to produce high quality documentation for a codebase in multiple formats (e.g., markdown, html, pdf).",

}

# Default temperature settings for different strategies
DEFAULT_TEMPERATURES = {
    "none": 0.7,
    "DAT_instructions": 0.7,  # Standard temperature for structured task
    "competitive": 0.7,
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