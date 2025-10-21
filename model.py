import torch.nn as nn

class TextGenerator(nn.Module):
    """
    The LSTM model architecture for text generation.
    This class defines the 'blueprint' for the AI's brain.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(TextGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. Embedding Layer: Converts word indices to dense vectors (256 dimensions).
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 2. LSTM Layer: The core memory unit. 
        # hidden_size=512 gives the model high capacity to learn complex patterns.
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)

        # 3. Final Output Layer: Maps LSTM output to vocabulary scores.
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        # Pass the sequence through the LSTM
        output, hidden = self.lstm(embedded, hidden)
        
        # We only need the prediction from the very last word in the sequence
        output = self.fc(output[:, -1, :])
        return output, hidden

    def init_hidden(self, batch_size, device):
        """
        Initializes the hidden state (h_0 and c_0) to all zeros.
        """
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device),
                weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device))
