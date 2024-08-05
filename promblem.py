# Let us see how to approach this problem. STAR, so we have a framework to look upto. Structuring helps.

## State the Assumptions:
"""
1. Working with gpt2-small SAEs, and the options are MLP section or attention(Focus on the MLP section this time).
2. SAEs are fixed, and we can must access them using SAELEns(we can go for any particular layer)
"""

### Questions we need to answer:
"""
1. Can we discover some feature importance and patterns in gpt2-small mlp section in a particular block?
2. What extra thing does MCTS provide that other methods have not exploited till now?
3. Why MCTS should not be used?
"""

## Test an initial version with core logic implmented:

## Use the test to produce some action:

## Use the result of the action to iterate.