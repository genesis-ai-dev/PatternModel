# ctrl + f for "FIXME" tags and to find what else needs to be added

import string
import os
import torch
from torch.utils.data import Dataset
from mingpt.model import GPT # Taken from Andrej Karpathy's MinGPT, thanks
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def rm(text):
    translation_table = str.maketrans({key: None for key in string.punctuation if key != ':'})
    return text.translate(translation_table).lower()

def get_config():
    C = CN()
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/reltokengpt'
    C.data = RelativeTokenDataset.get_default_config()
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt2'
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4
    C.trainer.batch_size = 4
    return C

class RelativeTokenizer:
    def __init__(self, context_size):
        self.context_size = context_size

    def tokenize(self, text):
        words = text.split()
        token_to_value = {'<EOP>': 0}
        value_to_token = {0: '<EOP>'}
        next_value = 1
        relative_tokens = []
        for word in words:
            if word not in token_to_value:
                token_to_value[word] = next_value
                value_to_token[next_value] = word
                next_value += 1
            relative_tokens.append(token_to_value[word])
        return relative_tokens, token_to_value, value_to_token

    def detokenize(self, relative_tokens, value_to_token):
        words = [value_to_token.get(value, 'unk') for value in relative_tokens]
        return ' '.join(words)

    def generate_input_output_pairs(self, text):
        input_output_pairs = []
        words = text.split()
        for i in range(len(words) - self.context_size):
            context = words[i:i + self.context_size]
            output_words = words[i + 1:i + self.context_size + 1]
            relative_tokens, token_to_value, value_to_token = self.tokenize(' '.join(context))
            input_seq = relative_tokens
            output_seq = [token_to_value.get(word, token_to_value['<EOP>']) for word in output_words]
            input_output_pairs.append((input_seq, output_seq))
        return input_output_pairs

class RelativeTokenDataset(Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 1000
        return C

    def __init__(self, config, data):
        self.config = config
        self.tokenizer = RelativeTokenizer(config.block_size)
        self.data = data
        self.words = data.split()

    def __len__(self):
        return len(self.words) - self.config.block_size

    def __getitem__(self, idx):
        context = self.words[idx:idx + self.config.block_size]
        output_words = self.words[idx + 1:idx + self.config.block_size + 1]
        relative_tokens, token_to_value, value_to_token = self.tokenizer.tokenize(' '.join(context))
        input_seq = relative_tokens
        output_seq = [token_to_value.get(word, token_to_value['<EOP>']) for word in output_words]
        x = torch.tensor(input_seq, dtype=torch.long)
        y = torch.tensor(output_seq, dtype=torch.long)
        return x, y

config = get_config()
print(config)
setup_logging(config)
set_seed(config.system.seed)

text = open('/content/data.txt', 'r').read()
text = rm(text)
train_dataset = RelativeTokenDataset(config.data, text)

config.model.vocab_size = config.data.block_size + 1
config.model.block_size = config.data.block_size
model = GPT(config.model)

trainer = Trainer(config.trainer, model, train_dataset)

def dif(text1, text2):
    matcher = SequenceMatcher(None, text1, text2)
    return matcher.ratio() * 100

history = []
diff_history = []

def batch_end_callback(trainer):
    if trainer.iter_num % 10 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        history.append(trainer.loss.item())
    if trainer.iter_num % 80 == 0:
        model.eval()
        with torch.no_grad():
            tokenizer = RelativeTokenizer(config.data.block_size)
            completion = context # FIXME, just define this somewhere
            for _ in range(900):
                relative_tokens, token_to_value, value_to_token = tokenizer.tokenize(completion)
                x = torch.tensor(relative_tokens, dtype=torch.long)[None, ...].to(trainer.device)
                logits = model(x)
                logits = logits[0] if isinstance(logits, tuple) else logits
                last_token_logits = logits[:, -1, :]
                last_token_probs = torch.softmax(last_token_logits, dim=-1)
                valid_token_values = list(value_to_token.keys())
                valid_token_probs = last_token_probs[:, valid_token_values]
                valid_token_probs /= valid_token_probs.sum(dim=-1, keepdim=True)
                next_token_value = torch.multinomial(valid_token_probs[0], num_samples=1).item()
                next_token = value_to_token[next_token_value]
                if next_token in ['<EOP>', '<eop>']:
                    break
                completion += ' ' + next_token.lower()
                if len(completion.split(' ')) > config.data.block_size - 100:
                    break
            completion = completion.split("\n")[-1].split("source:")[0]
            print(completion)
            print("target similarity: " + str(dif(completion, actual))) # FIXME: Also define 'actual' however
            diff_history.append(dif(completion, actual))
        ckpt_path = os.path.join(config.system.work_dir, "model.pt")
        torch.save(model.state_dict(), ckpt_path)
        model.train()

trainer.set_callback('on_batch_end', batch_end_callback)
trainer.run()

x = np.arange(len(diff_history))
plt.scatter(x, diff_history, label='Data Points')
m, b = np.polyfit(x, diff_history, 1)
plt.plot(x, m*x + b, color='red', label='Line of Best Fit')
plt.xlabel('Index')
plt.ylabel('Diff Value')
plt.title('Diff Values with Line of Best Fit')
plt.legend()
plt.show()

plt.plot(history)
plt.show()