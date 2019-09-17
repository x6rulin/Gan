import torch


class LogDloss(torch.nn.Module):

    def __init__(self):
        super(LogDloss, self).__init__()

        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, **kwargs):
        if 'fake_scores' in kwargs:
            if 'real_scores' in kwargs:
                return self.critic_loss(**kwargs)
            return self.generator_loss(**kwargs)

        raise RuntimeError("invalid arguments: expected {real_scores, fake_scores} or {fake_scores}")

    def critic_loss(self, real_scores, fake_scores):
        real_labels = torch.ones_like(real_scores, device=fake_scores.device)
        fake_labels = torch.zeros_like(fake_scores, device=fake_scores.device)

        real_loss = self.criterion(real_scores, real_labels)
        fake_loss = self.criterion(fake_scores, fake_labels)

        loss = real_loss + fake_loss
        return loss

    def generator_loss(self, fake_scores):
        real_labels = torch.ones_like(fake_scores, device=fake_scores.device)

        loss = self.criterion(fake_scores, real_labels)
        return loss


class EMLoss(torch.nn.Module):

    def forward(self, **kwargs):
        if 'fake_scores' in kwargs:
            if 'real_scores' in kwargs:
                return self.critic_loss(**kwargs)
            return self.generator_loss(**kwargs)

        raise RuntimeError("invalid arguments: expected {real_scores, fake_scores} or {fake_scores}")

    @staticmethod
    def critic_loss(real_scores, fake_scores):
        return (fake_scores - real_scores).mean()

    @staticmethod
    def generator_loss(fake_scores):
        return -fake_scores.mean()
