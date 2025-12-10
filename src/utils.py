def format_source_excerpt(text, max_length=300):
    """
    Format source text to be more readable.
    """
    # Basic cleaning
    text = ' '.join(text.split())  # Normalize whitespace
    
    # Truncate
    if len(text) > max_length:
        text = text[:max_length]
        # Find last complete sentence
        for punct in ['. ', '? ', '! ']:
            idx = text.rfind(punct)
            if idx > max_length * 0.6:
                text = text[:idx + 1]
                break
        text += "..."
    
    # Add paragraph breaks for readability
    sentences = text.split('. ')
    if len(sentences) > 3:
        # Break into paragraphs every 2-3 sentences
        formatted = []
        for i, sent in enumerate(sentences):
            formatted.append(sent.strip())
            if (i + 1) % 2 == 0 and i < len(sentences) - 1:
                formatted.append('\n')
        text = '. '.join(formatted)
    
    return text