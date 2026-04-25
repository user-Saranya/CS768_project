import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralDecisionTree(nn.Module):
    def __init__(self, input_dim, output_dim, depth=3):
        super().__init__()
        self.depth = depth
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Number of internal nodes and leaves
        self.num_internal = 2 ** depth - 1
        self.num_leaves = 2 ** depth

        # Internal node parameters (routing)
        self.decision = nn.Linear(input_dim, self.num_internal)

        # Leaf node values
        self.leaf_values = nn.Parameter(
            torch.randn(self.num_leaves, output_dim)
        )

    def forward(self, x):
        """
        x: [N, input_dim]
        returns: [N, output_dim]
        """

        batch_size = x.size(0)

        # Compute decision probabilities
        decision_logits = self.decision(x)  # [N, num_internal]
        decision_probs = torch.sigmoid(decision_logits)

        # Compute path probabilities to each leaf
        mu = x.new_ones(batch_size, 1)

        begin = 0
        end = 1

        for d in range(self.depth):
            nodes = decision_probs[:, begin:end]  # [N, 2^d]

            mu = mu.unsqueeze(-1)  # [N, 2^d, 1]

            # Left and right probabilities
            mu = torch.cat([mu * nodes.unsqueeze(-1),
                            mu * (1 - nodes.unsqueeze(-1))], dim=-1)

            mu = mu.view(batch_size, -1)

            begin = end
            end = begin + 2 ** (d + 1)

        # mu: [N, num_leaves]

        # Weighted sum of leaf values
        out = torch.matmul(mu, self.leaf_values)  # [N, output_dim]

        return out

class NeuralDecisionForest(nn.Module):
    def __init__(self, input_dim, output_dim, num_trees=5, depth=3):
        super().__init__()

        self.trees = nn.ModuleList([
            NeuralDecisionTree(input_dim, output_dim, depth)
            for _ in range(num_trees)
        ])

    def forward(self, x):
        outputs = [tree(x) for tree in self.trees]
        return torch.stack(outputs, dim=0).mean(dim=0)
