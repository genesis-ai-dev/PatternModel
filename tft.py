import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from relative_tokenizer import RelativeTokenizer
import torch.nn.functional as F

# Define the dataset class
class TextDataset(Dataset):
    def __init__(self, file_path, context_size):
        self.tokenizer = RelativeTokenizer(context_size)
        self.samples = self.load_data(file_path)
        self.tokenized_samples = self.tokenize_samples()

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            data = file.read()
        samples = data.split('<EOP>')
        return [sample.strip() for sample in samples if sample.strip()]

    def tokenize_samples(self):
        tokenized_samples = []
        for sample in self.samples:
            tokens, token_to_value, value_to_token = self.tokenizer.tokenize(sample)
            tokenized_samples.append((tokens, token_to_value, value_to_token))
        return tokenized_samples

    def __len__(self):
        return len(self.tokenized_samples)

    def __getitem__(self, idx):
        tokens, token_to_value, value_to_token = self.tokenized_samples[idx]
        return torch.tensor(tokens, dtype=torch.long), token_to_value, value_to_token

# Define the TFT model (from previous implementation)
class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        x = self.fc(x)
        return x[:, :, :x.shape[-1]//2] * torch.sigmoid(x[:, :, x.shape[-1]//2:])

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc3 = nn.Linear(input_dim, output_dim)
        self.glu = GLU(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        residual = self.fc3(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.glu(x)
        x = self.layer_norm(x + residual)
        return x

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VariableSelectionNetwork, self).__init__()
        self.gates = nn.Linear(input_dim, input_dim)
        self.grns = nn.ModuleList([GatedResidualNetwork(1, hidden_dim, 1) for _ in range(input_dim)])

    def forward(self, x):
        gates = torch.softmax(self.gates(x), dim=-1)
        outputs = torch.stack([g(x[..., i:i+1]) for i, g in enumerate(self.grns)], dim=-1)
        outputs = torch.sum(outputs * gates.unsqueeze(-1), dim=-2)
        return outputs

class TemporalFusionDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(TemporalFusionDecoder, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.glu = GLU(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = self.fc(x)
        attn_output, _ = self.attention(x, x, x)
        x = self.glu(attn_output)
        x = self.layer_norm(x + attn_output)
        return x

class TFT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, output_dim):
        super(TFT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.variable_selection = VariableSelectionNetwork(input_dim, hidden_dim)
        self.temporal_decoder = TemporalFusionDecoder(hidden_dim, hidden_dim, num_heads)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.variable_selection(x)
        x = self.temporal_decoder(x)
        x = self.output_layer(x)
        return x

# Load the data
file_path = 'synthetic3.txt'
context_size = 10
dataset = TextDataset(file_path, context_size)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model, loss function, and optimizer
input_dim = context_size
hidden_dim = 64
num_heads = 8
output_dim = 1

model = TFT(input_dim, hidden_dim, num_heads, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for tokens, _, _ in dataloader:
        tokens = tokens.permute(1, 0)  # Transpose to match expected input shape for TFT
        optimizer.zero_grad()
        output = model(tokens.float())
        loss = criterion(output, tokens.float())
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

print("Training complete.")
