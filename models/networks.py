import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FeedForwardNN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.activation = nn.GELU()
        self.norms = nn.ModuleList([nn.LayerNorm(size) for size in hidden_sizes])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.activation(self.norms[0](self.input_layer(x)))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.dropout(self.activation(self.norms[i+1](hidden_layer(x))))
        x = self.output_layer(x)
        return x
    
class ConceptBottleneckModel(nn.Module):
    def __init__(self, input_size, concept_hidden_sizes, concept_output_size, final_hidden_sizes, final_output_size):
        super(ConceptBottleneckModel, self).__init__()
        # Initialize concept_model with specified sizes
        self.concept_model = FeedForwardNN(input_size, concept_hidden_sizes, concept_output_size)
        # Initialize final_model using the output size of concept_model as its input size
        if not final_hidden_sizes:
            self.final_model = nn.Linear(concept_output_size, final_output_size)
        else:
            self.final_model = FeedForwardNN(concept_output_size, final_hidden_sizes, final_output_size)

    def forward(self, x):
        # Compute concept outputs
        concepts = self.concept_model(x)
        # Compute final outputs using concepts
        final_output = self.final_model(concepts)
        return concepts, final_output