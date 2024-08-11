# ----------------------- Imports ----------------------------

import torch
from sae_lens import SAE
from datasets import Dataset
from transformer_lens.utils import tokenize_and_concatenate
from transformer_lens import HookedTransformer

#------------------------- Prepare datasets for IOI tasks -------------------------

# Make these variables as well, so it'll be easier to plug and play.
SENTENCE = "When John and Marry went to the store, John gave a drink to "
OPTION1 = "John"
OPTION2 = "Mary"
CONTROL = "Adam"

## load the model and sae
device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained('gpt2-small', device = device) # do we even need this?

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = 'gpt2-small-res-jb',
    sae_id = "blocks.8.hook_resid_pre",
    device = device
)

#----------- Utility Functions -------------
def create_dataset(text):
    return Dataset.from_dict({"text": [text]})

def get_tokens(dataset, tokenizer=model.tokenizer, streaming=True, max_length=sae.cfg.context_size, add_bos_token=sae.cfg.prepend_bos):
    return tokenize_and_concatenate(
        dataset = dataset,
        tokenizer = tokenizer,
        streaming=streaming,
        max_length=max_length,
        add_bos_token=add_bos_token
    )

# -------------- This is the sentence we are going to use for IOI task. ------------------
sentence_dataset = create_dataset(SENTENCE)
sentence_tokens  = get_tokens(sentence_dataset)

# ------------- These are the options for the answers ------------------

option1_dataset = create_dataset(OPTION1)
option1_tokens  = get_tokens(option1_dataset)

option2_dataset = create_dataset(OPTION2)
option2_tokens  = get_tokens(option2_dataset)

control_dataset = create_dataset(CONTROL)
control_tokens  = get_tokens(control_dataset)

# --------------------------------------------------------------------


import math
import random
import numpy as np

## ----------------------------------------------------------------------------------------------------------------- ##
class MCTSNode:
    def __init__(self, state, parent=None) -> None:
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0

def ucb1(node):
    return node.value / node.visits + 2 * (2 * math.log(node.parent.visits) / node.visits) ** 0.5

def select(node):
    if not node.children:
        return node
    return select(max(node.children.values(), key=ucb1))

def expand(node, possible_actions):
    for action in possible_actions:
        if action not in node.children:
            new_state = apply_action(node.state, action)
            node.children[action] = MCTSNode(new_state, node)
    return random.choice(list(node.children.values())) # see if this is right. I can't understand right now.

def rollout(state, possible_actions, depth=5):
    total_reward = 0
    for _ in range(depth):
        action = random.choice(possible_actions)
        new_state = apply_action(state, action)
        reward = calculate_reward(new_state)
        
        total_reward += reward
        state = new_state
    return total_reward

def backpropogate(node, result):
    while node:
        node.visits += 1
        node.value += result
        node = node.parent
        
def MCTS(root, possible_actions, num_simulations=100):
    print(f"Starting MCTS loop. Number of simulations = {num_simulations}")
    for i in range(num_simulations):
        node = select(root)
        print(f"It. = {i+1}. Select")
        child = expand(node, possible_actions)
        print(f"It. = {i+1}. Expand")
        result = rollout(child.state, possible_actions)
        print(f"It. = {i+1}. Rollout")
        backpropogate(child, result)
        print(f"It. = {i+1}. Backpropagate")
    
    return max(root.children, key=lambda c: root.children[c].visits)

## ----------------------------------------------------------------------------------------------------------------- ##
        
def apply_action(state, action):
    # For a given state, it performs the action and gives a new state in return.
    return action     # I think for the first experiment, this is right.

# Q: Why are we returning the action without any chages? Can `state + action` work? 
# A: In the calculate_reward function, we are calculating similarity based on the vector missing word and sentence without the missing word. This actually makes sense.


def calculate_reward(original_state, new_state):
    # Based on some "shift" between the original and the new state, we calculate and return reward. 
    # For now, we'll go with simple similarity score.  
    similarity = np.dot(original_state, new_state) / (np.linalg.norm(original_state) * np.linalg.norm(new_state))
    return similarity

## ----------------------------------------------------------------------------------------------------------------- ##

def run_ioi_experiment(num_samples, num_simulations, expanded_shape):
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