import torch
import torch.nn as nn
from torch.func import vmap


class GaussianBinaryRBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim, sigma = 0.1):
        super(GaussianBinaryRBM, self).__init__()
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


    def contrastive_divergence(self, Batch_data, k=1, lr=0.3):
        """ Implémente l'algorithme de CD """
        Batch_data = Batch_data.to(self.W.device)  

        v0 = Batch_data.detach()  # On détache du graphe de calcul

        # Phase positive
        h0, P_h_given_v0 = self.sample_h(v0)

        # Gibbs Sampling (k étapes)
        hk = h0
        for _ in range(k):
            vk = self.sample_v(hk)
            hk, P_h_given_vk = self.sample_h(vk)

        # Mise à jour des poids sur tout le batch
        with torch.no_grad():
            self.h_bias += lr * (P_h_given_v0 - P_h_given_vk).mean(dim=0)
            self.v_bias += lr * (v0 - vk).mean(dim=0)
            self.W += lr * (torch.matmul(P_h_given_v0.T, v0) - torch.matmul(P_h_given_vk.T, vk)) / Batch_data.shape[0]



    def persistent_contrastive_divergence(self, batch_data, persistent_v = None, k=1, lr=0.1):
        """ Implémente l'algorithme de PCD """
        v0 = batch_data
        v0 = v0.to(self.W.device)
        
        # Initialisation des chaînes persistantes si nécessaire
        if persistent_v is None:
            persistent_v = v0.clone().detach()
        
        # Phase positive (avec les données d'entraînement)
        h0, P_h_given_v0 = self.sample_h(v0)
        
        # Phase négative (avec les échantillons persistants)
        vk = self.persistent_v  # Utilisation des échantillons précédents
        for _ in range(k):
            hk, P_h_given_vk = self.sample_h(vk)
            vk = self.sample_v(hk)
        
        # Mise à jour des poids et biais
        with torch.no_grad():
            self.h_bias += lr * (P_h_given_v0 - P_h_given_vk).mean(dim=0)
            self.v_bias += lr * (v0 - vk).mean(dim=0)
            self.W += lr * (torch.matmul(P_h_given_v0.T, v0) - torch.matmul(hk.T, vk)) / v0.shape[0]
        
        # Mise à jour des chaînes persistantes
        persistent_v = vk.detach()
        return persistent_v


    def forward(self, Batch_data, iter=1):
        """Passe avant récursif appliqué à un batch de données"""
        for _ in range(iter):
            h, _ = self.sample_h(Batch_data)  # Échantillonner pour la couche cachée
            Batch_data_pred = self.sample_v(h)  # Ré-échantillonner pour la couche visible
        return Batch_data_pred

      


