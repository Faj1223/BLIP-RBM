import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim):
        super(RBM, self).__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        # Poids de connexion entre couches visibles et cachées
        self.W = nn.Parameter(torch.randn(hidden_dim, visible_dim) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(visible_dim))  # Biais visibles
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim))  # Biais cachés

    def sample_h(self, v):
        """Échantillonne la couche cachée à partir des visibles"""
        p_h_given_v = torch.sigmoid(torch.matmul(v, self.W.T) + self.h_bias)
        return torch.bernoulli(p_h_given_v)

    def sample_v(self, h):
        """Échantillonne la couche visible à partir des cachées"""
        p_v_given_h = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return torch.bernoulli(p_v_given_h)

    def contrastive_divergence(self, v0, k=1, lr=0.01):
        """Applique le Contraste de Divergence CD-k pour l'apprentissage"""
        v = v0.detach()
        
        # Propagation avant
        h0 = self.sample_h(v)
        
        # Gibbs Sampling (k étapes)
        for _ in range(k):
            v = self.sample_v(h0)
            h0 = self.sample_h(v)

        # Mise à jour des poids
        h0_v0 = torch.matmul(h0.T, v0)
        h_k_v_k = torch.matmul(h0.T, v)

        self.W.data += lr * (h0_v0 - h_k_v_k)
        self.v_bias.data += lr * torch.mean(v0 - v, dim=0)
        self.h_bias.data += lr * torch.mean(h0 - h0, dim=0)

    def forward(self, v):
        """Passe avant simple (encode les données)"""
        h = self.sample_h(v)
        return h  # Les embeddings raffinés
