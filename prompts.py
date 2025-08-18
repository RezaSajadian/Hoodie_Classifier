"""
Prompt templates for hoodie classification.
Includes both plain prompts and chain-of-thought reasoning prompts.
"""

TWO_PIECE_PROMPTS = [
    "a hoodie with a 2-piece hood",
    "a hoodie where the hood is made of two mirrored panels",
    "a hoodie with a hood that has a single center seam",
    "a hoodie with a hood made from two parts joined together",
    "a hoodie with a simple two-panel hood design",
    "This is a photo of a hoodie. A 2-piece hoodie has a hood made from two mirrored panels joined by a single center seam. The image shows a hood with two panels. Therefore it is a 2-piece hoodie.",
    "This is a photo of a hoodie. Looking at the hood construction, I can see it's made from two separate panels that are joined together at the center. This indicates a 2-piece hoodie design.",
    "This is a photo of a hoodie. The hood appears to be constructed from two main pieces that meet at the center, creating a single seam line. This is characteristic of a 2-piece hoodie."
]

THREE_PIECE_PROMPTS = [
    "a hoodie with a 3-piece hood",
    "a hoodie where the hood has three parts: two side panels and one center gusset",
    "a hoodie with a hood that has a center gusset running from front to back",
    "a hoodie with a hood made from three separate pieces",
    "a hoodie with a hood that includes a center panel between two side panels",
    "This is a photo of a hoodie. A 3-piece hoodie has a center gusset running from front to back and two side panels. The image shows a center gusset. Therefore it is a 3-piece hoodie.",
    "This is a photo of a hoodie. Looking at the hood construction, I can see it's made from three parts: two side panels and a center gusset panel. This indicates a 3-piece hoodie design.",
    "This is a photo of a hoodie. The hood appears to have three distinct sections - two side panels and a center gusset that runs from the front to the back. This is characteristic of a 3-piece hoodie."
]

def get_all_prompts():
    """Get all prompts for both classes."""
    return {
        "2": TWO_PIECE_PROMPTS,
        "3": THREE_PIECE_PROMPTS
    }
