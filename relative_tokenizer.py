

class RelativeTokenizer:
    def __init__(self, context_size):
        self.context_size = context_size

    def tokenize(self, text):
        tokens = list(text) # Split the text into words
        token_to_value = {'<EOP>': 0}  # Assign 0 to the <EOP> (end of pattern) token
        value_to_token = {0: '<EOP>'}  # Assign <EOP> to the value 0
        next_value = 1  # Start assigning values from 1
        relative_tokens = []
        for token in tokens:
            if token not in token_to_value:
                token_to_value[token] = next_value
                value_to_token[next_value] = token
                next_value += 1
            relative_tokens.append(token_to_value[token])
        return relative_tokens, token_to_value, value_to_token

    def detokenize(self, relative_tokens, value_to_token):
        tokens = [value_to_token[value] for value in relative_tokens]
        return ''.join(tokens)  # Join the words with spaces