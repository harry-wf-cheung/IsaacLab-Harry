# import torch
# import gpytorch
# from matplotlib import pyplot as plt
# from torch.utils.data import DataLoader, TensorDataset

# torch.manual_seed(0)

# # ==== Data ====
# import h5py
# import torch
# from torch.utils.data import DataLoader

# import robomimic.utils.file_utils as FileUtils


# # Load data from HDF5 file
# with h5py.File('./docs/training_data/varying_beaker_scale_split.hdf5', 'r') as file:
    

# # Create dataset and dataloader
# train_dataset = TensorDataset(train_x, train_y)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# # Now you can keep the rest of your code the same, replacing your synthetic data with these tensors.

# # train_x = torch.linspace(-3, 3, 100).unsqueeze(-1)
# # train_y = torch.sin(train_x * (3.0)) + 0.3 * torch.randn(train_x.size())
# # test_x = torch.linspace(-3, 3, 200).unsqueeze(-1)

# # ==== GP Layer ====
# class GP_Layer(gpytorch.models.ApproximateGP):
#     def __init__(self, input_dim, inducing_points):
#         variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(0))
#         variational_strategy = gpytorch.variational.VariationalStrategy(
#             self, inducing_points, variational_distribution, learn_inducing_locations=True
#         )
#         super().__init__(variational_strategy)

#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
#         )

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# # ==== Deep GP Model ====
# class DeepGPModel(gpytorch.models.ApproximateGP):
#     def __init__(self, input_dim):
#         # Main strategy is for the last layer
#         inducing_points = torch.randn(64, input_dim)
#         variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(64)
#         variational_strategy = gpytorch.variational.VariationalStrategy(
#             self, inducing_points, variational_distribution, learn_inducing_locations=True
#         )
#         super().__init__(variational_strategy)

#         # Hidden layer
#         self.hidden_layer = GP_Layer(input_dim, inducing_points=torch.randn(64, input_dim))

#         # Output mean/cov modules
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
#         )

#     def forward(self, x):
#         hidden_out = self.hidden_layer(x).rsample()
#         mean = self.mean_module(hidden_out)
#         covar = self.covar_module(hidden_out)
#         return gpytorch.distributions.MultivariateNormal(mean, covar)

# # ==== Setup ====
# model = DeepGPModel(input_dim=1)
# likelihood = gpytorch.likelihoods.GaussianLikelihood()

# model.train()
# likelihood.train()

# mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
# optimizer = torch.optim.Adam([
#     {'params': model.parameters()},
#     {'params': likelihood.parameters()},
# ], lr=0.01)

# train_dataset = TensorDataset(train_x, train_y)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# # ==== Training ====
# epochs = 200
# for epoch in range(epochs):
#     total_loss = 0
#     for x_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         output = model(x_batch)
#         loss = -mll(output, y_batch.squeeze())
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     if epoch % 20 == 0 or epoch == epochs - 1:
#         print(f"Epoch {epoch:3d} - Loss: {total_loss:.3f}")

# # ==== Evaluation ====
# model.eval()
# likelihood.eval()
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     preds = likelihood(model(test_x))
#     mean = preds.mean
#     lower, upper = preds.confidence_region()

# # ==== Plot ====
# plt.figure(figsize=(10, 5))
# plt.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Train Data')
# plt.plot(test_x.numpy(), mean.numpy(), 'b', label='Mean')
# plt.fill_between(test_x.squeeze(), lower, upper, alpha=0.5, label='Confidence')
# plt.legend()
# plt.title("Deep GP Regression with Uncertainty")
# plt.savefig("./docs/training_data/gp.png")


import torch
import gpytorch
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
import os
import robomimic.utils.file_utils as FileUtils
import torch.utils.data as data_utils

# For reproducibility (matches train.py)
torch.manual_seed(0)
np.random.seed(0)

# === Load Dataset from HDF5, similar to train.py ===
def load_dataset_from_hdf5(hdf5_path, obs_key='obs', action_key='actions'):
    """
    Loads observation and action data from HDF5 into torch tensors.

    Args:
        hdf5_path (str): path to the dataset
        obs_key (str): HDF5 key for observations (default 'obs')
        action_key (str): HDF5 key for actions (default 'actions')

    Returns:
        train_x (torch.Tensor): observations
        train_y (torch.Tensor): actions
    """
    with h5py.File(hdf5_path, 'r') as f:
        obs_list = []
        act_list = []
        data_group = f['data']
        mask_group = f['mask']
        for data_group_key in data_group.keys():
            print(data_group_key)
            print(np.array(data_group[data_group_key]['actions'])[:5][0])
            exit()   

    return train_x, train_y

# === GP Model Definition (as before) ===
class GP_Layer(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, inducing_points):
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DeepGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim):
        # Main strategy for last layer
        inducing_points = torch.randn(64, input_dim)
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(64)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        # Hidden layer
        self.hidden_layer = GP_Layer(input_dim, inducing_points=torch.randn(64, input_dim))

        # Output mean/cov modules
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        )

    def forward(self, x):
        hidden_out = self.hidden_layer(x).rsample()
        mean = self.mean_module(hidden_out)
        covar = self.covar_module(hidden_out)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

# === Training Loop ===
def train_gp_model(hdf5_path, epochs=200, batch_size=64, learning_rate=0.01, device='cpu'):
    # Load dataset
    train_x, train_y = load_dataset_from_hdf5(hdf5_path)

    # Move data to device
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # Create dataset and loader
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate model and likelihood
    input_dim = train_x.shape[1] if len(train_x.shape) > 1 else 1
    model = DeepGPModel(input_dim=input_dim).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    model.train()
    likelihood.train()

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} - Loss: {total_loss:.3f}")

    return model, likelihood

# === Main ===
if __name__ == "__main__":
    # Adjust path to your actual dataset location and keys inside HDF5 if needed
    dataset_path = './docs/training_data/varying_beaker_scale_split.hdf5'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, likelihood = train_gp_model(dataset_path, device=device)
