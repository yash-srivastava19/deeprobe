# ----------------------- Imports ----------------------------

import math
import einops 
import torch
import random
import numpy as np
from sae_lens import SAE
from datasets import Dataset
from datasets.load import load_dataset
from transformer_lens import HookedTransformer

#------------------------- Prepare datasets for IOI tasks -------------------------

## load the model and sae
device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained('gpt2-small', device = device) # do we even need this?

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = 'gpt2-small-res-jb',
    sae_id = "blocks.8.hook_resid_pre",
    device = device
)

#----------- Utility Functions -------------

def keep_single_column(dataset, col_name: str):
    """
    Acts on a HuggingFace dataset to delete all columns apart from a single column name - useful when we want to tokenize and mix together different strings
    """
    for key in dataset.features:
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset

def token_concat(
    dataset,
    tokenizer,
    streaming: bool = False,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
    num_proc: int = 10,
):
    """Helper function to tokenizer and concatenate a dataset of text. This converts the text to tokens, concatenates them (separated by EOS tokens) and then reshapes them into a 2D array of shape (____, sequence_length), dropping the last batch. Tokenizers are much faster if parallelised, so we chop the string into 20, feed it into the tokenizer, in parallel with padding, then remove padding at the end.

    This tokenization is useful for training language models, as it allows us to efficiently train on a large corpus of text of varying lengths (without, eg, a lot of truncation or padding). Further, for models with absolute positional encodings, this avoids privileging early tokens (eg, news articles often begin with CNN, and models may learn to use early positional encodings to predict these)

    Args:
        dataset (Dataset): The dataset to tokenize, assumed to be a HuggingFace text dataset.
        tokenizer (AutoTokenizer): The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        streaming (bool, optional): Whether the dataset is being streamed. If True, avoids using parallelism. Defaults to False.
        max_length (int, optional): The length of the context window of the sequence. Defaults to 1024.
        column_name (str, optional): The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token (bool, optional): . Defaults to True.

    Returns:
        Dataset: Returns the tokenized dataset, as a dataset of tensors, with a single column called "tokens"

    Note: There is a bug when inputting very small datasets (eg, <1 batch per process) where it just outputs nothing. I'm not super sure why
    """
    dataset = keep_single_column(dataset, column_name)
    if tokenizer.pad_token is None:
        # We add a padding token, purely to implement the tokenizer. This will be removed before inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # Define the length to chop things up into - leaving space for a bos_token if required
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length

    def tokenize_function(examples):
        # Taken from TransformerLens, but without the chunking.
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by eos_tokens
        full_text = tokenizer.eos_token.join(text)
        
        #FIX: Here we moved away from the reference implementation.
        tokens = tokenizer(full_text, return_tensors="np", padding=True)["input_ids"].flatten()
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]
        
        num_tokens = len(tokens)
        num_batches = num_tokens // (seq_len)
        
        
        # Drop the final tokens if not enough to make a full sequence
        tokens = tokens[: seq_len * num_batches] if num_batches else tokens
        
        if add_bos_token:
            if num_batches:
                tokens = einops.rearrange(tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len)
                prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
                tokens = np.concatenate([prefix, tokens], axis=1)
            else:
                tokens = np.array(tokens)
        return {"tokens": tokens}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=(num_proc if not streaming else None),
        remove_columns=[column_name],
    )
    tokenized_dataset.set_format(type="torch", columns=["tokens"])
    return tokenized_dataset

# DRY principles. Make a function to perform these tasks.

SENTENCE_SLUG = "ioi_sentence"
OPTION1_SLUG = "ioi_option1"
OPTION2_SLUG = "ioi_option2"
CONTROL_SLUG = "ioi_option3"


def create_dataset(slug):
    ds = load_dataset(
        path=f"yash-srivastava19/{slug}",
        split="train",
        streaming=False,
    )
    return ds

def get_tokens(dataset, tokenizer=model.tokenizer, streaming=True, max_length=sae.cfg.context_size, add_bos_token=sae.cfg.prepend_bos):
    return token_concat(
        dataset = dataset,
        tokenizer = tokenizer,
        streaming=streaming,
        max_length=max_length,
        add_bos_token=add_bos_token
    )


# -------------- This is the sentence we are going to use for IOI task. ------------------

sentence_dataset = create_dataset(SENTENCE_SLUG)
sentence_tokens  = get_tokens(sentence_dataset)

# ------------- These are the options for the answers ------------------

option1_dataset = create_dataset(OPTION1_SLUG)
option1_tokens  = get_tokens(option1_dataset)

option2_dataset = create_dataset(OPTION2_SLUG)
option2_tokens  = get_tokens(option2_dataset)

control_dataset = create_dataset(CONTROL_SLUG)
control_tokens  = get_tokens(control_dataset)


## ----------------------------------------------------------------------------------------------------------------- ##
possible_actions = [i for i in range(sae.cfg.context_size)] # select one out of context size.

## ----------------------------------------------------------------------------------------------------------------- ##
class MCTSNode:
    def __init__(self, state, parent=None) -> None:
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0

def ucb1(node):
    if node.visits:
        val = node.value / node.visits + 2 * (2 * math.log(node.parent.visits) / node.visits) ** 0.5
        #print(f"Val = {val}")
        return val # This should just work.
    #print(f"Val(no visits) = {1e16}")
    return 1e16  # any big number works

def select(node):
    if not node.children : # if there are no children, return the node.
        #print(node.children.values())
        return node
    #print(node.children.values())
    return select(max(node.children.values(), key=ucb1))

def expand(node, possible_actions):
    for action in possible_actions:
        if action not in node.children:
            new_state = apply_action(node.state, action)
            node.children[action] = MCTSNode(new_state, node)
    return random.choice(list(node.children.values())) # see if this is right. I can't understand right now.

def rollout(state, depth=5):
    total_reward = 0
    for _ in range(depth):
        action = random.choice(possible_actions)
        new_state = apply_action(state, action)
        reward = calculate_reward(state, new_state)
        
        total_reward += reward
        state = new_state
    return total_reward

def backpropogate(node, result):
    while node:
        node.visits += 1
        node.value += max(result)
        node = node.parent
        
def MCTS(root, possible_actions, num_simulations=10):
    print(f"Starting MCTS loop. Number of simulations = {num_simulations}")
    for i in range(num_simulations):
        node = select(root)
        print(f"It. = {i+1}. Select")
        child = expand(node, possible_actions)
        print(f"It. = {i+1}. Expand")
        result = rollout(child.state)
        print(f"It. = {i+1}. Rollout")
        backpropogate(child, result)
        print(f"It. = {i+1}. Backpropagate")
    
    return max(root.children, key=lambda c: root.children[c].visits)


## ----------------------------------------------------------------------------------------------------------------- ##
        
def apply_action(state, action):
    # For a given state, it performs the action and gives a new state in return.
    return state     # I think for the first experiment, this is right.

# Q: Why are we returning the action without any chages? Can `state + action` work? 
# A: In the calculate_reward function, we are calculating similarity based on the vector missing word and sentence without the missing word. This actually makes sense.


def calculate_reward(original_state, new_state):
    # Based on some "shift" between the original and the new state, we calculate and return reward. 
    # For now, we'll go with simple similarity score.  
    similarity = (original_state @ new_state.T) / (np.linalg.norm(original_state) * np.linalg.norm(new_state))
    return similarity

## ----------------------------------------------------------------------------------------------------------------- ##

def run_ioi_experiment(num_samples=3, num_simulations=5):
    # We'll try to make sense of the IOI task. The original state is the embedding of a sentence geenrally used for IOI taks.
    # The possible actions include the number of people in the sentence + 1 control object to see whether the reward we are getting is correct or not. 
    
    
    # First, we create the original state. This is the original sentence tokens, expanded to SAE latent dim.     
    original_state = sentence_tokens["tokens"]  # based on the latent space dims for SAE.
    
    X = [option1_tokens["tokens"], option2_tokens["tokens"], control_tokens["tokens"]] # List of token ids for SAE activations for the possible actions .
    
    # once we have all of these, we can start the MCTS loop
    print("Starting the MCTS Loop.")
    
    for i in range(num_samples):
        root = MCTSNode(original_state) # make the original state as the root of the Tree.
        print(f"Created the root node for OBJECT {i}")
        
        best_action = MCTS(root, X, num_simulations) # Then, based on the token-ids of the options, get the best action.
        print(f"Found the best action for the given OBJECT : {i}")
        
        new_state = apply_action(original_state, best_action) # apply the best action to get the new state.
        print(f"Based on the best action, we calcualte the new state for the OBJECT : {i}")
        
        reward = calculate_reward(original_state, new_state) # based 
        print(f"Reward for OBJECT {i}: {reward}")