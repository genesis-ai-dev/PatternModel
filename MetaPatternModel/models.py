"""models.py

Contains a `GlobalRelativeTokenizer` which is a slight modification to RelativeTokenizer.
- Both GPT4MetaTranslator and ClaudeMetaTranslator use it to convert target sequences into ints 
where each unique word is assigned an int. The model (Claude, GPT-4o, etc..) is then tasked with
figuring out which source sequences map to target sequences (this is now language agnostic because of the tokenization)
Then, it is given a new source sequence and told to convert it to a similar list of ints, the result is then detokenized back into the target language. 
- This should force pattern matching and work well regardless of the language.
"""
import string
from anthropic import Client
from openai import OpenAI


class GlobalRelativeTokenizer:
    def __init__(self):
        """
        Initializes the tokenizer with default values and tokens.
        """
        self.context_size = 500
        self.token_to_value = {'<EOP>': 0}  # Assign 0 to the <EOP> token
        self.value_to_token = {0: '<EOP>'}  # Assign <EOP> to the value 0
        self.next_value = 1  # Start assigning values from 1

    def tokenize(self, text):
        """
        Tokenizes the given text into numerical values based on unique words.

        Parameters:
        text (str): The text to be tokenized.

        Returns:
        list: A list of numerical tokens representing the text.
        """
        tokens = text.split(" ")
        relative_tokens = []
        for token in tokens:
            if token not in self.token_to_value:
                self.token_to_value[token] = self.next_value
                self.value_to_token[self.next_value] = token
                self.next_value += 1
            relative_tokens.append(self.token_to_value[token])
        return relative_tokens

    def detokenize(self, relative_tokens):
        """
        Converts numerical tokens back into text.

        Parameters:
        relative_tokens (list): A list of numerical tokens.

        Returns:
        str: The detokenized text.
        """
        tokens = [self.value_to_token[value] for value in relative_tokens]
        return ' '.join(tokens)

def filter_text(text):
    """
    Filters and cleans the input text by converting to lowercase and removing punctuation.

    Parameters:
    text (str): The text to be filtered.

    Returns:
    str: The filtered text.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation.replace(":", '').replace("\\", "")))
    return text



class GPT4MetaTranslator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initializes the translator with API key and model.

        Parameters:
        api_key (str): The API key for accessing the translation service.
        model (str): The model name to be used for translation.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.tokenizer = GlobalRelativeTokenizer()

    def translate(self, input_prompt: str, pairs: str):
        """
        Translates the input prompt based on the provided translation pairs.

        Parameters:
        input_prompt (str): The source sequence to be translated.
        pairs (str): The translation pairs for reference.

        Returns:
        str: The translated output.
        """
        pairs, input_prompt = self.preprocess(pairs, input_prompt)
        system = SYSTEM_PROMPT
        message = MESSAGE_PROMPT.format(pairs=pairs, input_prompt=input_prompt)
        output = self.send_message(message, system)
        return output

    def send_message(self, message, system):
        """
        Sends a message to the translation service and retrieves the output.

        Parameters:
        message (str): The message to be sent.
        system (str): The system prompt.

        Returns:
        str: The output from the translation service.
        """
        response = self.client.chat.completions.create(model=self.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": message}
        ],
        temperature=0.01,
        max_tokens=2048)
        output = response.choices[0].message.content
        print(output)
        return output

    def preprocess(self, pairs, input_prompt):
        """
        Preprocesses the translation pairs and input prompt for translation.

        Parameters:
        pairs (str): The translation pairs.
        input_prompt (str): The source sequence to be translated.

        Returns:
        tuple: Processed pairs and input prompt.
        """
        lines = pairs.split("\n")
        new_pairs = []
        for line in lines:
            if line.startswith("source:"):
                new_pairs.append(line)
            elif line.startswith("target:"):
                target_text = filter_text(line[len("target: "):])
                tokenized_target = 'target: ' + ' '.join(map(str, self.tokenizer.tokenize(target_text)))
                new_pairs.append(tokenized_target)
        new_pairs = "\n".join(new_pairs)
        return new_pairs, input_prompt

class ClaudeMetaTranslator:
    def __init__(self, key: str, model: str = "claude-3-opus-20240229"):
        """
        Initializes the translator with API key and model.

        Parameters:
        key (str): The API key for accessing the translation service.
        model (str): The model name to be used for translation.
        """
        self.client = Client(api_key=key)
        self.model = model
        self.tokenizer = GlobalRelativeTokenizer()

    def translate(self, input_prompt: str, pairs: str):
        """
        Translates the input prompt based on the provided translation pairs.

        Parameters:
        input_prompt (str): The source sequence to be translated.
        pairs (str): The translation pairs for reference.

        Returns:
        str: The translated output.
        """
        pairs, input_prompt = self.preprocess(pairs, input_prompt)
        system = SYSTEM_PROMPT
        message = MESSAGE_PROMPT.format(pairs=pairs, input_prompt=input_prompt)
        output = self.send_message(message, system)
        return output

    def send_message(self, message, system):
        """
        Sends a message to the translation service and retrieves the output.

        Parameters:
        message (str): The message to be sent.
        system (str): The system prompt.

        Returns:
        str: The output from the translation service.
        """
        chat_completion = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[
                {"role": "user", "content": message}
            ],
            temperature=0.01,
            max_tokens=2048
        )
        output = chat_completion.content[0].text
        print(output)
        return output

    def preprocess(self, pairs, input_prompt):
        """
        Preprocesses the translation pairs and input prompt for translation.

        Parameters:
        pairs (str): The translation pairs.
        input_prompt (str): The source sequence to be translated.

        Returns:
        tuple: Processed pairs and input prompt.
        """
        lines = pairs.split("\n")
        new_pairs = []
        for line in lines:
            if line.startswith("source:"):
                new_pairs.append(line)
            elif line.startswith("target:"):
                target_text = filter_text(line[len("target: "):])
                tokenized_target = 'target: ' + ' '.join(map(str, self.tokenizer.tokenize(target_text)))
                new_pairs.append(tokenized_target)
        new_pairs = "\n".join(new_pairs)
        return new_pairs, input_prompt

def extract_numbers(text):
    """
    Extracts numerical tokens from the given text.

    Parameters:
    text (str): The text containing numerical tokens.

    Returns:
    list: A list of numerical tokens extracted from the text.
    """
    import re
    pattern = r'target:\s*(\d+(?:\s+\d+)*)'
    matches = re.findall(pattern, text)
    if matches:
        last_match = matches[-1]
        return last_match.split()
    return []

# Constants for prompts
SYSTEM_PROMPT = """You are a Translator, skilled at finding general sequence-to-sequence patterns in data. In this case, there are 'source' and 'target' sequences.
I'll provide you with some translation pairs and your task is to generate the 'target' sequence corresponding to the given 'source' sequence.
These are language translation pairings, but each unique word in the target has been assigned a number. Thus, the overall patterns should be linguistic in nature.
Take a deep breath, clear your mind, and think this through step by step, outlining your evidence for each decision.
"""

MESSAGE_PROMPT = """Based on the provided translation pairs and your own intelligence, generate the 'target' sequence corresponding to the given 'source' sequence.

1. First, create a 'target word bank' of target numbers that will probably occur given the source sequence. Write out the probable target numbers, and say why you think each is probable. Essentially create a mapping of source phrases to target sequences.
2. Formulate grammatical rules that seem apparent in the target texts, and are relevant to the sequence you will translate.
3. Next, focus on arranging these target numbers in the correct order to form a coherent translation, also say why.
4. Remember: don't copy any of the examples verbatim, just use the underlying linguistic rules that seem apparent.
5. Avoid repetition.
Here are the translation pairs:
6. Only predict tokens that have already been used.
Here are the pairs:
{pairs}

Source Sequence:
{input_prompt}

Remember, the last part of your response should follow this format 'target: <some output>'
Think through each step, write out your thought process, be concise, and use the examples provided.
Do your best, don't stress about it, use the format. My friend doesn't think you can do this well, but I am fully confident you can prove him wrong.
"""
