# Deeprobe - Probing In Expanded Feature Space

I have been thinking a lot about how to use Sparse Autoencoders and MCTS to feature discovery, and Deeprobe is exactly that. 
We'll look step by step on what exactly are we tring to do, and in between we'll come up with hypothesis/things to test.

## Intermediate Progress

- [] Started studying LLMs with MCTS
- [] Too broad, and focused on improving results, so started the studying GANs and MCTS
- [] Too vague, found paper with stepwise evaluation, similar to Q*, so moved aways as I don't want to explore that area right now.
- [] Started with Autoencoders and MCTS, and found paper that generated molecules using this, and it was interesting.
- [] With all the little pieces in mind, went to Claude and asked about potential use cases(Autoencoders + MCTS), and it said working in compressed latent space is easier for simulation.
- [] In expanded latent space, we can use the feature map + MCTS to "make better decisions".
- [] Other than that, we can also do feature importance and pattern recognition, which I'm exploring using a genomic data example. 