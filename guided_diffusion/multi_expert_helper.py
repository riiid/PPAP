import torch.nn as nn

class MultiExpertWrapper(nn.Module):
    def __init__(self, models, n_experts, num_timesteps):
        super().__init__()
        self.models = models
        self.n_experts = n_experts
        self.num_timesteps = num_timesteps
        self.step_range = [ int(i * self.num_timesteps / self.n_experts) for i in range(0, self.n_experts + 1)]

    def step_to_classifier_index(self, t):
        classifier_steps = self.step_range
        for i in range(0, len(classifier_steps)):
            if classifier_steps[i] <= t[0] <= classifier_steps[i + 1]:
                return i

    def forward(self, x, t):
        index = self.step_to_classifier_index(t)
        y = self.models[index](x)
        return y