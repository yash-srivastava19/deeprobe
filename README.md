# Deeprobe - Probing In Expanded Feature Space

I have been thinking a lot about how to use Sparse Autoencoders and MCTS to feature discovery, and Deeprobe is exactly that. 
We'll look step by step on what exactly are we tring to do, and in between we'll come up with hypothesis/things to test.

## Deeprobe

Feature discovery and pattern recognition in SAEs is a difficult problem. There are many ways to discover important features in SAE, but I tried using MCTS for IOI task. Here's what I found.

- Turns out, we can understand feature importance on IOI task using MCTS.
- Although there are pitfalls when working with SAE and MCTS(due to expanded feature space), having a solid reward policy might help us.

### Experimemt
IOI(Indirect Object Identification) task is one of the tasks against which model interpretability methods are tested. The task goes something like this :

For a given sentence like "When John and Marry went to the store, John gave a drink to ..." 
We have to prove that the feature "score" of Mary should be the highest. This is a difficult for LMs to solve. This is how we are doing it :

- First, take a sentence similar to the one given in the example.
- The list of possible actions will then be the number of objects in the sentence, plus one control object to measure reward baseline.
- For the given number of samples, we do MCTS on particular activations of the possible actions and see which reward is the highest(which, in the most simple case is the similarity score).
- Compare the results to see whether MCTS can be used to find the best possible answer for IOI task

## Further Work
Update: At least the initial version of the code works. You can do basic MCTS on SAEs, but what exactly we are getting from it is still being worked on.


## Intermediate Progress

- [x] Started studying LLMs with MCTS
- [x] Too broad, and focused on improving results, so started the studying GANs and MCTS
- [x] Too vague, found paper with stepwise evaluation, similar to Q*, so moved aways as I don't want to explore that area right now.
- [x] Started with Autoencoders and MCTS, and found paper that generated molecules using this, and it was interesting.
- [x] With all the little pieces in mind, went to Claude and asked about potential use cases(Autoencoders + MCTS), and it said working in compressed latent space is easier for simulation.
- [x] In expanded latent space, we can use the feature map + MCTS to "make better decisions".
- [x] Other than that, we can also do feature importance and pattern recognition, which I'm exploring using a genomic data example. 
- [x] Desgined an experiment to study feature importance in IOI task.
- [x] Got the IOI data from HF, and used a subset of it to make and load dataset.
- [x] There was a problem with the `tokenize_and_concatenate` function in Transformer Lens utils for small datasets, so fixed a local patch for it.
- [x] Tried running some experiments, and with all the assumtions we have, it is working, although there are a lot of caveats.
- [] Need to understand how exactly we can leverage full powers of MCTS.