# This file is just placeholder, experiments and all will be perfomred in Kaggle notebooks.
import torch

from sae_lens import SAE
from transformer_lens import HookedTransformer

## load the model and sae
device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained('gpt2-small', device = device) # do we even need this?

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = 'gpt2-small-res-jb',
    sae_id = "blocks.8.hook_resid_pre",
    device = device
)

## load and process the dataset. Do we even need it?

# dataset = load_dataset(
#     path = "NeelNanda/pile-10k",
#     split = "train",
#     streaming=False
# )

# token_dataset = tokenize_and_concatenate(
#     dataset = dataset,
#     tokenizer = model.tokenizer,
#     streaming = True, 
#     max_length = sae.cfg.context_size,
#     add_bos_token = sae.cfg.prepend_bos
# )
