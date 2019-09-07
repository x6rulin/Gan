import torch


class LogDloss(torch.nn.Module):

    def __init__(self, device):
        super(LogDloss, self).__init__()

        self.device = device
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, **kwargs):
        if 'fake_scores' in kwargs:
            if 'real_scores' in kwargs:
                return self.critic_loss(**kwargs)
            return self.generator_loss(**kwargs)

        raise RuntimeError("invalid arguments: expected [real_scores, fake_scores] or [fake_scores]")

    def critic_loss(self, real_scores, fake_scores):
        real_labels = torch.ones_like(real_scores, device=self.device)
        fake_labels = torch.ones_like(fake_scores, device=self.device)

        real_loss = self.criterion(real_scores, real_labels)
        fake_loss = self.criterion(fake_scores, fake_labels)

        loss = real_loss + fake_loss
        return loss

    def generator_loss(self, fake_scores):
        real_labels = torch.ones_like(fake_scores, device=self.device)

        loss = self.criterion(fake_scores, real_labels)
        return loss
