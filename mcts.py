""" There are 4 parts of MCTS: Tree traversal, Node Expansion, Rollout and Backprop"""

import math
import random

## The inputs we need

possible_actions = None # We will set this based on SAE. It will be a list of all possible actions
X = None # We will set this based on the SAE(expanded features)

## Utility functions

def apply_action(state, action):
    # Action is a tuple of (feature_idx, operation)
    new_state = state.copy()
    # here, based on the operation, we will change the value of the feature_idx
    return new_state

def calculate_reward(state):
    # This is a simple sum reward. Reality can be more complex
    return sum(state)


def find_correlated_features(X, threshold):
    patterns = set()
    # based on the correlation matrix, do some traveral and add to patterns
    return patterns

## MCTS stuff
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
    return random.choice(list(node.children.values())) # see if this is right

def rollout(state, depth=5):
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
    for _ in range(num_simulations):
        node = select(root)
        child = expand(node, possible_actions)
        result = rollout(child.state)
        backpropogate(child, result)
    
    return max(root.children, key=lambda c: root.children[c].visits)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_simulations", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--expanded_shape", type=int, default=10)
    args = parser.parse_args()

    feature_imp = [0] * args.expanded_shape

    for i in range(args.num_samples):
        root = MCTSNode(X[i])
        best_action = MCTS(root, possible_actions, args.num_simulations)
        feature_imp[best_action[0]] += 1

    patterns = find_correlated_features(X, 0.5)
    for i, (f1, f2) in enumerate(patterns):
        print("Pattern {}: {} and {} are highly correlated".format(i, f1, f2))