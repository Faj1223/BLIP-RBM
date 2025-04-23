import torch
import torch.nn as nn
from torch.func import vmap
import matplotlib.pyplot as plt


class GaussianBinaryRBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim, sigma=0.1, lr_f=0.1, weight_decay_f=0.01, gamma = 0.01, alpha=0.8):
        super(GaussianBinaryRBM, self).__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma

        # Device unique pour tout
        Temp = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Température sur le bon device et transformée
        self.T = torch.tensor(Temp, device=self.device)

        # Paramètres principaux
        self.W = nn.Parameter(torch.randn(hidden_dim, visible_dim, device=self.device) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(visible_dim, device=self.device))
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim, device=self.device))

        self.lr_f = lr_f
        self.weight_decay_f = weight_decay_f

        # Paramètres rapides, initialisés avec les mêmes tailles que les params principaux
        self.W_f = torch.zeros_like(self.W, device=self.device)
        self.h_bias_f = torch.zeros_like(self.h_bias, device=self.device)
        self.v_bias_f = torch.zeros_like(self.v_bias, device=self.device)


    def sample_h(self, v, T):
        """Échantillonne la couche cachée à partir des visibles"""
        p_h_given_v = torch.sigmoid((torch.matmul(v, self.W.T) + self.h_bias) / T)
        return torch.bernoulli(p_h_given_v), p_h_given_v

    def sample_v(self, h):
        """Échantillonne la couche visible à partir des cachées (distribution gaussienne)"""
        mean_v_given_h = torch.matmul(h, self.W) + self.v_bias
        v_sample = mean_v_given_h + self.sigma * torch.randn_like(mean_v_given_h)
        return torch.tanh(v_sample)

    def recuit_simule(self, W_update, W_old):
        delta_W = W_update - W_old
        correction = (W_old * delta_W).sum()
        T_new = self.T + self.gamma * (1.0 / self.T) * correction
        return T_new


    def contrastive_divergence(self, Batch_data, k=1, lr=0.3):
        """Implémente l'algorithme de CD avec recuit simulé"""
        Batch_data = Batch_data.to(self.W.device)
        v0 = Batch_data.detach()

        T = self.T  # Température actuelle
        h0, P_h_given_v0 = self.sample_h(v0, T)

        hk =  h0
        for _ in range(k):
            vk = self.sample_v(hk)
            vk = self.alpha * v0 + (1 - self.alpha) * vk
            hk, P_h_given_vk = self.sample_h(vk, T)

        # Phase de mise à jour des poids
        with torch.no_grad():
            W_old = self.W.clone().detach()
            W_new = W_old.clone()

            self.h_bias += lr * (P_h_given_v0 - P_h_given_vk).mean(dim=0)
            self.v_bias += lr * (v0 - vk).mean(dim=0)
            W_new += lr * (torch.matmul(P_h_given_v0.T, v0) - torch.matmul(P_h_given_vk.T, vk)) / Batch_data.shape[0]

            W_new = W_new.clone().detach()
            self.T = self.recuit_simule(W_new, W_old)
            self.W = W_new





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
        vk = persistent_v  # Utilisation des échantillons précédents
        for _ in range(k):
            hk, P_h_given_vk = self.sample_h(vk)
            vk = self.sample_v(hk)
            vk = self.alpha * v0 + (1 - self.alpha) * vk # rappelle
        
        # Mise à jour des poids et biais
        with torch.no_grad():
            self.h_bias += lr * (P_h_given_v0 - P_h_given_vk).mean(dim=0)
            self.v_bias += lr * (v0 - vk).mean(dim=0)
            self.W += lr * (torch.matmul(P_h_given_v0.T, v0) - torch.matmul(hk.T, vk)) / v0.shape[0]
        
        # Mise à jour des chaînes persistantes
        persistent_v = vk.detach()
        return persistent_v
    

    def fast_persistent_contrastive_divergence(self, batch_data, persistent_v=None, k=1, lr=0.1):
        batch_data = batch_data.to(self.W.device)
        
        # Initialisation des échantillons persistants
        if persistent_v is None:
            persistent_v = batch_data.clone().detach()
        
        # Phase positive
        h0, P_h_given_v0 = self.sample_h(batch_data)
        
        # Phase négative (avec échantillons persistants et fast parameters)
        vk = persistent_v
        for _ in range(k):
            hk, P_h_given_vk = self.sample_h(vk)
            vk = self.sample_v(hk)
            vk = self.alpha * batch_data + (1 - self.alpha) * vk # rappelle
        
        # Mise à jour des poids du modèle principal
        with torch.no_grad():
            self.h_bias += lr * (P_h_given_v0 - P_h_given_vk).mean(dim=0)
            self.v_bias += lr * (batch_data - vk).mean(dim=0)
            self.W += lr * (torch.matmul(P_h_given_v0.T, batch_data) - torch.matmul(P_h_given_vk.T, vk)) / batch_data.shape[0]
            
            # Mise à jour des paramètres rapides
            self.W_f += self.lr_f * ((torch.matmul(P_h_given_v0.T, batch_data) - torch.matmul(P_h_given_vk.T, vk)) / batch_data.shape[0])
            self.h_bias_f += self.lr_f * (P_h_given_v0 - P_h_given_vk).mean(dim=0)
            self.v_bias_f += self.lr_f * (batch_data - vk).mean(dim=0)
            
            # Appliquer la décroissance des paramètres rapides
            self.W_f *= (1 - self.weight_decay_f)
            self.h_bias_f *= (1 - self.weight_decay_f)
            self.v_bias_f *= (1 - self.weight_decay_f)
        
        # Mise à jour des chaînes persistantes
        persistent_v = vk.detach()
        return persistent_v
    

    def train(self, data, epochs=300):
        """Entraîne le modèle RBM avec recuit simulé sur plusieurs epochs"""

        self.T_history = []
        for epoch in range(epochs):
            # Shuffle des données
            idx = torch.randperm(data.size(0))
            shuffled_data = data[idx]

            # Applique le contraste de divergence
            self.contrastive_divergence(shuffled_data)
            self.T_history.append(self.T.item())

            # Évolution de la température
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} - Température actuelle : {self.T:.4f}")

        plt.plot(self.T_history)
        plt.title("Évolution de la température")
        plt.xlabel("Itération")
        plt.ylabel("Température T")
        plt.grid(True)
        plt.show()
   



    def forward(self, Batch_data, iter=1):
        """Passe avant récursif appliqué à un batch de données"""
        for _ in range(iter):
            h, _ = self.sample_h(Batch_data)  # Échantillonner pour la couche cachée
            Batch_data_pred = self.sample_v(h)  # Ré-échantillonner pour la couche visible
        return Batch_data_pred

      


