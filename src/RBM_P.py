import torch
import torch.nn as nn
from torch.func import vmap

class GaussianBinaryRBM_P(nn.Module):
    def __init__(self, visible_dim, hidden_dim, sigma=0.1):
        super(GaussianBinaryRBM_P, self).__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.sigma = sigma

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Poids de connexion entre les couches visibles et cachées
        self.W = nn.Parameter(torch.randn(hidden_dim, visible_dim, device=device) * 0.01)
        # Biais
        self.v_bias = nn.Parameter(torch.zeros(visible_dim, device=device))  # Biais visibles
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim, device=device))   # Biais cachés
    
    def sample_h(self, v):
        """Parallélisation du calcul de h_i sachant v"""
        p_h_given_v = torch.sigmoid(torch.matmul(v, self.W.T) + self.h_bias)
        
        # Échantillonnage en parallèle sur chaque h_i
        h_sample = vmap(torch.bernoulli, randomness="different")(p_h_given_v)
        
        return h_sample, p_h_given_v

    def sample_v(self, h):
        """Parallélisation du calcul de v_i sachant h"""
        mean_v_given_h = torch.matmul(h, self.W) + self.v_bias
        noise = self.sigma * torch.randn_like(mean_v_given_h)  # Ajout de bruit gaussien

        # Échantillonnage en parallèle sur chaque v_i
        v_sample = vmap(torch.tanh)(mean_v_given_h + noise)

        return v_sample

    def contrastive_divergence(self, Batch_data, k=1, lr=0.3):
        Batch_data = Batch_data.float().to(self.W.device)
        v0 = Batch_data.detach()  # On détache du graphe de calcul

        # Phase positive
        h0, P_h_given_v0 = self.sample_h(v0)

        # Gibbs Sampling optimisé
        vk, hk = vmap(lambda h: self.gibbs_sampling(h, k), in_dims=0, randomness="different")(h0)

        # Mise à jour des poids en mode sécurisé
        with torch.no_grad():
            self.W += lr * (torch.matmul(P_h_given_v0.T, v0) - torch.matmul(self.sample_h(vk)[1].T, vk)) / Batch_data.shape[0]
            self.v_bias += lr * (v0 - vk).mean(dim=0)
            self.h_bias += lr * (P_h_given_v0 - self.sample_h(vk)[1]).mean(dim=0)

    # Fonction Gibbs Sampling parallèle
    def gibbs_sampling(self, h, k):
        for _ in range(k):
            v = self.sample_v(h)
            h, _ = self.sample_h(v)
        return v, h

    def forward(self, Batch_data, iter=1):
        """Passe avant récursif appliqué à un batch de données"""
        for _ in range(iter):
            h, _ = self.sample_h(Batch_data)  # Échantillonner pour la couche cachée
            Batch_data_pred = self.sample_v(h)  # Ré-échantillonner pour la couche visible
        return Batch_data_pred
