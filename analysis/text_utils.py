from better_profanity import profanity

# Load profanity word list once on module import
profanity.load_censor_words()

def mask_swearwords(text):
    """
    Masks profanities in the given text using asterisks.
    Example: 'shit' → 's***'
    """
    return profanity.censor(text)
