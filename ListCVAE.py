import torch
import torch.utils.data
from torch import nn
import numpy as np
from utils import default_device

class Parameters:
    def __init__(self, batch_norm, dropout, activation_function):
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation_function = activation_function


class ListCVAE(nn.Module):
    def __init__(self, num_documents, slate_size, response_dims, embed_dims, encoder_dims, latent_dims, decoder_dims,
                 prior_dims, device):
        super(ListCVAE, self).__init__()
        self.device = device

        self.num_of_documents = num_documents
        self.slate_size = slate_size
        self.response_dims = response_dims
        self.embed_dims = embed_dims
        self.encoder_dims = encoder_dims
        self.latent_dims = latent_dims

        # Index work from 0 - (num_of_movies - 1). Thus, we use num_of_movies as a padding index
        self.padding_idx = self.num_of_documents

        # +1 for the padding index
        self.embedding_documents = torch.nn.Embedding(num_embeddings=self.num_of_documents + 1,
                                                      embedding_dim=self.embed_dims,
                                                      padding_idx=self.padding_idx)

        # Encoder
        layers_block = []
        input_dims = (self.embed_dims * self.slate_size) + self.response_dims

        for out_dims in encoder_dims:
            layers_block.extend(self.encoder_block(input_dims, out_dims))
            input_dims = out_dims

        self.encoder_layers = nn.Sequential(
            *layers_block
        )

        self.encoder_mu = nn.Linear(input_dims, self.latent_dims)
        self.encoder_log_variance = nn.Linear(input_dims, self.latent_dims)

        # Decoder
        layers_block = []
        input_dims = self.latent_dims + self.response_dims

        for out_dims in decoder_dims:
            layers_block.extend(self.decoder_block(input_dims, out_dims))
            input_dims = out_dims

        self.decoder_layers = nn.Sequential(
            *layers_block,
            nn.Linear(input_dims, self.embed_dims * self.slate_size)
        )

        # Prior
        layers_block = []
        input_dims = self.response_dims

        for out_dims in prior_dims:
            layers_block.extend(self.prior_block(input_dims, out_dims))
            input_dims = out_dims

        self.prior_layers = nn.Sequential(
            *layers_block
        )

        self.prior_mu = nn.Linear(input_dims, self.latent_dims)
        self.prior_log_variance = nn.Linear(input_dims, self.latent_dims)

    @staticmethod
    def encoder_block(in_feat, out_feat):
        block = [nn.Linear(in_feat, out_feat)]
        return block

    @staticmethod
    def decoder_block(in_feat, out_feat):
        block = [nn.Linear(in_feat, out_feat)]
        # block.append(nn.Sigmoid())
        return block

    @staticmethod
    def prior_block(in_feat, out_feat):
        block = [nn.Linear(in_feat, out_feat)]
        # block.append(nn.Sigmoid())
        return block

    def encode(self, slate_inputs, conditioned_info):
        # Slate Embeds
        slate_embeds = self.embedding_documents(slate_inputs)
        slate_embeds = slate_embeds.view(slate_embeds.shape[0], self.slate_size * self.embed_dims)

        encoder_input = torch.cat((slate_embeds, conditioned_info), dim=1)

        out = self.encoder_layers(encoder_input)

        last_hidden = out

        mu = self.encoder_mu(last_hidden)
        log_variance = self.encoder_log_variance(last_hidden)

        return mu, log_variance, last_hidden

    def reparameterize(self, mu, log_variance):
        """
        https://stats.stackexchange.com/questions/199605/how-does-the-reparameterization-trick-for-vaes-work-and-why-is-it-important
        :param mu:
        :param log_variance:
        :return:
        """
        std = torch.exp(0.5 * log_variance)
        eps = torch.rand_like(std, device=self.device)

        return mu + eps * std

    def decode(self, z, conditioned_info):
        decoder_input = torch.cat((z, conditioned_info), dim=1)

        out = self.decoder_layers(decoder_input)

        all_documents = torch.arange(self.num_of_documents, device=self.device)
        all_document_embeddings = self.embedding_documents(all_documents).T

        out = out.view(out.shape[0], self.slate_size, self.embed_dims)

        out = torch.matmul(out, all_document_embeddings)

        # out = nn.Softmax(dim=-1)(out)
        out = nn.Sigmoid()(out)

        return out

    def forward(self, slate_input, response_vector):
        conditioned_info = response_vector

        # Encoder
        mu, log_variance, last_hidden_real = self.encode(slate_input, conditioned_info)

        # Decoder
        z = self.reparameterize(mu, log_variance)
        decoder_out = self.decode(z, conditioned_info)

        # Prior
        # prior_mu = torch.tensor(np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]],
        #                                                       z.shape[0])).to(default_device())
        # prior_log_variance = torch.tensor(np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]],
        #                                                                 z.shape[0])).to(default_device())

        # prior_out = self.prior_layers(conditioned_info)
        # prior_mu = self.prior_mu(prior_out)
        # prior_log_variance = self.prior_log_variance(prior_out)

        # prior_mu = torch.zeros((z.shape[0], 2)).to(default_device())
        # prior_log_variance = torch.ones((z.shape[0], 2)).to(default_device())

        return decoder_out, mu, log_variance, last_hidden_real

    def get_slates(self, user_interactions_with_padding, decoder_out):
        slates = []
        masking = torch.zeros([decoder_out.shape[0], decoder_out.shape[2]], device=self.device, dtype=torch.float32)
        masking = masking.scatter_(1, user_interactions_with_padding, float('-inf'))

        for slate_item in range(self.slate_size):
            slate_output = decoder_out[:, slate_item, :]
            slate_output = slate_output + masking
            slate_item = torch.argmax(slate_output, dim=1)

            slates.append(slate_item)
            masking = masking.scatter_(1, slate_item.unsqueeze(dim=1), float('-inf'))

        return torch.stack(slates, dim=1)

    def inference(self, response_vector):
        # Personalized
        # movie_embedding = self.embedding_movies(user_interactions_with_padding)

        # conditioned_info = torch.cat((user_embedding, response_vector), dim=1)
        conditioned_info = response_vector

        # Prior
        # prior_out = self.prior_layers(conditioned_info)
        # prior_mu = self.prior_mu(prior_out)
        # prior_log_variance = self.prior_log_variance(prior_out)

        # prior_mu = torch.zeros((conditioned_info.shape[0], 2)).to(default_device())
        # prior_log_variance = torch.ones((conditioned_info.shape[0], 2)).to(default_device())

        # z = self.reparameterize(prior_mu, prior_log_variance)

        z = np.random.normal(size=[conditioned_info.shape[0], self.latent_dims])
        z = torch.tensor(z, dtype=torch.float).to(default_device())

        decoder_out = self.decode(z, conditioned_info)

        return z, decoder_out
