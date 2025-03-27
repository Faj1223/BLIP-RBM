import torch
import torch.nn as nn

class GaussianBinaryRBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim, sigma = 0.1):
        super(GaussianBinaryRBM, self).__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.sigma = sigma

        # Poids de connexion entre les couches visibles et cachées
        self.W = nn.Parameter(torch.randn(hidden_dim, visible_dim) * 0.01)
        # Biais
        self.v_bias = nn.Parameter(torch.zeros(visible_dim))  # Biais visibles
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim))  # Biais cachés

    def sample_h(self, v):
        """Échantillonne la couche cachée à partir des visibles"""
        p_h_given_v = torch.sigmoid(torch.matmul(v, self.W.T) + self.h_bias)
        return torch.bernoulli(p_h_given_v), p_h_given_v

    def sample_v(self, h):
        """Échantillonne la couche visible à partir des cachées (distribution gaussienne)"""
        # On suppose que les visibles sont normalement distribuées, on échantillonne à partir de la normale
        mean_v_given_h = torch.matmul(h, self.W) + self.v_bias
        # Loi normale centrée en mean_v_given_h et d'écart type sigma
        v_sample = mean_v_given_h + self.sigma*torch.randn_like(mean_v_given_h) 

        # Normalisation dans [-1, 1] avec tanh
        return torch.tanh(v_sample)

    def contrastive_divergence(self, v0, k=1, lr=0.01):
        """Applique le Contraste de Divergence CD-k pour l'apprentissage"""
        v = v0.detach()
    
        # Propagation avant
        h0, P_h_given_v0 = self.sample_h(v)  # phase positive
    
        # Gibbs Sampling k étapes (phases négatives)
        hk = h0
        for _ in range(k):
            vk = self.sample_v(hk)
            hk, P_h_given_vk = self.sample_h(vk)
    
        # Mise à jour des poids de manière vectorisée
        delta_h_bias = P_h_given_v0 - P_h_given_vk
        self.h_bias.data += lr * delta_h_bias

        delta_v_bias = v0 - vk
        self.v_bias.data += lr * delta_v_bias

        delta_W = torch.outer(delta_h_bias, v0) - torch.outer(delta_h_bias, vk)
        self.W.data += lr * delta_W

                

        
        
    def forward(self, v, iter=1):
        """Passe avant récursif pour raffiner les embeddings"""
        if iter <= 0:
            return v  # Arrêt de la récursion
    
        h,_ = self.sample_h(v)
        v = self.sample_v(h)
    
        return self.forward(v, iter - 1)  # Appel récursif avec iter - 1

