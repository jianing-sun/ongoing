import numpy as np
from ListCVAE import ListCVAE, Parameters
import torch
from torch.utils.data import Dataset
from utils import default_device
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class DataSimulator(Dataset):
    def __init__(self, n=100, k=10, u=1, sigma=0.5, num_samples=10):
        self.num_documents = n
        self.slate_size = k
        self.sigma = sigma
        self.u = u
        self.num_samples = num_samples
        self.W = np.random.normal(self.u, self.sigma,
                                  (self.slate_size, self.num_documents, self.slate_size, self.num_documents))
        self.A = np.random.uniform(0, 1, self.num_documents)
        self.slates, self.ratings, self.probs = self.generate_samples()

    def generate_samples(self):
        slates, ratings, prob_ratings = [], [], []
        for _ in range(self.num_samples):
            slate = np.random.choice(self.num_documents, self.slate_size)
            rating = []
            prob_rating = []
            for i in range(self.slate_size):
                di = slate[i]
                j = np.arange(i + 1)
                dj = slate[j]
                W_ = self.W[i, slate[i], j, dj]
                p_ = self.A[di] * np.prod(W_)
                p_ = max(0, min(1, p_))
                ri = np.random.binomial(1, p_)
                rating.append(ri)
                prob_rating.append(p_)
            slates.append(slate.tolist())
            ratings.append(rating)
            prob_ratings.append(prob_rating)

        return torch.tensor(slates), torch.tensor(ratings), torch.tensor(prob_ratings)

    def __len__(self):
        return len(self.slates)

    def __getitem__(self, idx):
        s = self.slates[idx]
        r = self.ratings[idx]
        p = self.probs[idx]
        return s, r, p


class ExperimentListCVAE(nn.Module):
    def __init__(self, model, data_class):
        super(ExperimentListCVAE, self).__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.data_class = data_class
        self.KL_weight = None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.001)

    def CVAE_loss(self, slates_pred, slates_true, mu, log_variance, prior_mu, prior_log_variance, epoch):
        slates_pred = slates_pred.view(slates_pred.shape[0] * slates_pred.shape[1], slates_pred.shape[2])
        slates_true = slates_true.view(slates_true.shape[0] * slates_true.shape[1])

        entropy_loss = self.criterion(slates_pred, slates_true)

        mean_term = ((mu - prior_mu) ** 2) / prior_log_variance.exp()

        KL = 0.5 * torch.sum(prior_log_variance - log_variance + (log_variance.exp() / prior_log_variance.exp())
                             + mean_term - 1)

        return (KL * self.KL_weight[epoch]) + entropy_loss if self.KL_weight else KL + entropy_loss

    def CVAE_loss_normal_dist_prior(self,  slates_pred, slates_true, mu, log_variance, epoch):
        slates_pred = slates_pred.view(slates_pred.shape[0] * slates_pred.shape[1], slates_pred.shape[2])
        slates_true = slates_true.view(slates_true.shape[0] * slates_true.shape[1])

        entropy_loss = self.criterion(slates_pred, slates_true)

        mean_term = mu ** 2

        KL = 0.5 * torch.sum(log_variance.exp() + mean_term - 1 - log_variance)

        KL_loss = (KL * self.KL_weight[epoch]) if self.KL_weight else KL
        recon_loss = entropy_loss

        return KL_loss, recon_loss

    def run_experiment(self, num_epochs=100):
        for epoch in range(num_epochs):
            train_kl_loss, train_recon_loss = self.run_training_epoch(epoch)
            print(epoch, train_kl_loss, train_recon_loss)
            if (epoch+1) % 10 == 0:
                expectations, Z = self.run_inference()
                Z = torch.cat(Z, 0)
                expectations = torch.cat(expectations, 0)
                self.plot(Z.detach().cpu().numpy(), expectations.detach().cpu().numpy())

    def plot(self, Z, expectations):
        plt.scatter(Z[:, 0], Z[:, 1], c=expectations, cmap=plt.cm.get_cmap('jet', 10))
        plt.colorbar()
        plt.show()

    def run_inference(self):
        expectations = []
        Z = []
        with torch.no_grad():
            for batch, (batch_slates, batch_response, batch_prob_rating) in enumerate(self.data_class):
                batch_slates = batch_slates.to(default_device()).long()
                batch_response = batch_response.to(default_device()).float()
                batch_prob_rating = batch_prob_rating.to(default_device()).float()

                click_vectors = torch.full((batch_slates.shape[0], batch_slates.shape[1]),
                                          1, device=default_device(), dtype=torch.float32)
                response_vectors_opt = click_vectors.sum(dim=1).unsqueeze(dim=1)

                z, decoder_out = self.model.inference(response_vectors_opt)
                expectation = self.expect_num_clicks(batch_response, batch_prob_rating, decoder_out)
                expectations.append(expectation)
                Z.append(z)
        return expectations, Z

    def expect_num_clicks(self, ratings, prob_rating, decoder_out):
        # slate_input = slate_input.view(-1, slate_input.shape[-1], 1)
        softmax_out = decoder_out.detach().cpu().numpy()
        pred_slates = []
        for i, probs in enumerate(softmax_out):
            pred_slates.append([np.random.choice(100, 1, x.tolist())[0] for x in probs])
        pred_slates = torch.tensor(pred_slates).to(default_device())
        p_s = torch.stack([decoder_out[x, i, z] for x, y in zip(range(len(decoder_out[:, 0])), pred_slates)
                           for i, z in enumerate(y)])
        p_s = p_s.view(len(decoder_out[:, 0]), pred_slates.shape[1], 1)
        expectation = ratings.unsqueeze(-1) * prob_rating.unsqueeze(-1) * p_s
        expectation = expectation.squeeze(-1).sum(1)
        return expectation

    def run_training_epoch(self, epoch):
        self.model.train()
        kl_losses, recon_losses, expectations = [], [], []

        for batch, (batch_slates, batch_response, _) in enumerate(self.data_class):
            self.optimizer.zero_grad()

            batch_slates = batch_slates.to(default_device()).long()
            batch_response = batch_response.to(default_device()).float()
            # batch_prob_rating = batch_prob_rating.to(default_device()).float()

            response_vectors = batch_response.sum(dim=1).unsqueeze(dim=1)
            decoder_out, mu, log_variance, last_hidden_real = self.model(batch_slates, response_vectors)

            kl_loss, recon_loss = self.CVAE_loss_normal_dist_prior(decoder_out, batch_slates, mu, log_variance, epoch)

            # expectation = self.expect_num_clicks(batch_response, batch_prob_rating, decoder_out)

            (kl_loss + recon_loss).backward()
            self.optimizer.step()

            kl_losses.append(float(kl_loss))
            recon_losses.append(float(recon_loss))
            # expectations.append(expectation.mean())

        # print(torch.stack(expectations).mean().detach().cpu().numpy())

        return np.mean(kl_losses), np.mean(recon_losses)


if __name__ == "__main__":

    data_simulator = DataSimulator(n=1000, num_samples=1000)
    data_simulator.generate_samples()
    data_class = DataLoader(data_simulator, batch_size=100, shuffle=True, num_workers=1,
                            drop_last=True)

    for idx, (slate, response, p_r) in enumerate(data_simulator):
        print(idx, slate, response, p_r)

    model = ListCVAE(num_documents=1000,
                     slate_size=10,
                     response_dims=1,
                     embed_dims=8,
                     encoder_dims=[128, 128],
                     latent_dims=2,
                     decoder_dims=[128, 128],
                     prior_dims=[16, 32],
                     device=default_device()).to(default_device())

    print(model)

    experiment_class = ExperimentListCVAE(model, data_class)
    experiment_class.run_experiment(500)
