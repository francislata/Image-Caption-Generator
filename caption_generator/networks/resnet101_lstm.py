"""This contains the ResNet101 and LSTM network for the encoder-decoder network."""

from torchvision.models import resnet101
import torch.nn as nn

class ResNet101LSTM(nn.Module):
    """This subclass creates a network with ResNet101 and LSTM."""
    def __init__(self, vocab, **kwargs):
        super(ResNet101LSTM, self).__init__()

        self.vocab = vocab
        self._create_modules(**kwargs)

    def forward(self, x, y):
        # Encode the image into a fixed size vector of features
        img_features = self.encoder(x)
        img_features = img_features.view(img_features.size(0), -1)

        # Run the image features through an LSTM cell to get the hidden state
        h_state, c_state = self.lstm_cell(img_features)
        h_state, c_state = h_state.unsqueeze(0), h_state.unsqueeze(0)

        # Perform teacher-forcing for training the network
        y = self.emb(y)
        y, (h_state, c_state) = self.lstm(y, (h_state, c_state))

        return y

    def _create_modules(self,
                        pretrained,
                        num_img_features,
                        embedding_dim,
                        hidden_size,
                        bidirectional=False,
                        num_layers=1,
                        dropout=0):
        """Creates the modules for this network."""
        modified_encoder = list(resnet101(pretrained=pretrained).children())[:-1]

        self.encoder = nn.Sequential(*modified_encoder)
        self.emb = nn.Embedding(len(self.vocab.stoi), embedding_dim)
        self.lstm_cell = nn.LSTMCell(num_img_features, hidden_size)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
            dropout=dropout
        )
