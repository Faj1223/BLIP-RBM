import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class GaussianBinaryRBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim, sigma=0.1, alpha=0.73, initial_temp=10.0, gamma=0.18):
        super(GaussianBinaryRBM, self).__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Température initiale
        self.T = initial_temp

        # Paramètres du modèle
        self.W = nn.Parameter(torch.randn(hidden_dim, visible_dim, device=self.device) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(visible_dim, device=self.device))
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim, device=self.device))

        self.T_history = []

    def sample_h(self, v):
        activation = (torch.matmul(v, self.W.T) + self.h_bias) / self.T
        prob = torch.sigmoid(activation)
        return torch.bernoulli(prob), prob

    def sample_v(self, h):
        mean = torch.matmul(h, self.W) + self.v_bias
        return torch.tanh(mean + self.sigma * torch.randn_like(mean))

    def energy(self, v, h):
        term1 = ((v - self.v_bias) ** 2).sum(dim=1) / (2 * self.sigma ** 2)
        term2 = -(h @ self.h_bias)
        term3 = -(h @ self.W @ v.T).diag()
        return (term1 + term2 + term3).mean()

    def simulated_annealing_update(self, current_energy, prev_energy):
        delta = current_energy - prev_energy
        acceptance_prob = torch.exp(-delta / self.T) if delta > 0 else 1.0
        if torch.rand(1).item() < acceptance_prob:
            self.T *= self.gamma  # accepte et refroidit
        else:
            self.T *= 1.05  #rechaufement modéré

    def contrastive_divergence(self, data, k=1, lr=0.1):
        data = data.to(self.device)
        v0 = data.detach()
        h0, p_h0 = self.sample_h(v0)

        v = v0.clone()
        for _ in range(k):
            h, _ = self.sample_h(v)
            v = self.sample_v(h)
            v = self.alpha * v0 + (1 - self.alpha) * v

        h_end, p_hk = self.sample_h(v)

        # Recuit simulé 
        energy_initial = self.energy(v0, h0)
        energy_final = self.energy(v, h_end)
        self.simulated_annealing_update(energy_final, energy_initial)

        with torch.no_grad():
            # Update des poids
            self.W += lr * ((p_h0.T @ v0) - (p_hk.T @ v)) / data.size(0)

            # Mise à jour spéciale RBM gaussienne-binaire
            v_bias_initial = self.v_bias.clone().detach()
            self.v_bias += lr * (-(v0 - v_bias_initial) + (v - self.v_bias)).mean(dim=0)

            # Update du biais caché (inchangé)
            self.h_bias += lr * (p_h0 - p_hk).mean(dim=0)


    

    def train(self, data, batch_size=100, epochs=100):
        data = data.to(self.device)
        n_samples = data.size(0)
        self.energy_history = []
        self.T_history = []

        for epoch in range(epochs):
            perm = torch.randperm(n_samples)
            epoch_energy = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch = data[perm[i:i + batch_size]]
                self.contrastive_divergence(batch)

                # Calcul de l'énergie après mise à jour (optionnel : avant aussi)
                h, _ = self.sample_h(batch)
                energy = self.energy(batch, h).item()
                epoch_energy += energy
                n_batches += 1

            avg_energy = epoch_energy / n_batches
            self.energy_history.append(avg_energy)
            self.T_history.append(self.T)

            print(f"Epoch {epoch+1}/{epochs} | Température: {self.T:.4f} | Énergie moyenne: {avg_energy:.4f}")

        # Affichage des courbes après l'entraînement
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.T_history)
        plt.title("Évolution de la température")
        plt.xlabel("Epoch")
        plt.ylabel("T")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.energy_history)
        plt.title("Évolution de l'énergie moyenne")
        plt.xlabel("Epoch")
        plt.ylabel("Énergie")
        plt.grid(True)

        plt.tight_layout()
        plt.show()



    def forward(self, Batch_data, iter=1):
        """Passe avant récursif appliqué à un batch de données"""
        for _ in range(iter):
            h, _ = self.sample_h(Batch_data)  # Échantillonner pour la couche cachée
            Batch_data_pred = self.sample_v(h)  # Ré-échantillonner pour la couche visible
        return Batch_data_pred

