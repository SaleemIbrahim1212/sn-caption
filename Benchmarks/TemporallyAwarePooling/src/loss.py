import torch
import torch.nn.functional as F


class NLLLoss(torch.nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, labels, output):
        # return torch.mean(labels * -torch.log(output) + (1 - labels) * -torch.log(1 - output))

        return torch.mean(torch.mean(labels * -torch.log(output) + (1 - labels) * -torch.log(1 - output)))

class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
    def forward(self, model_vids, model_text):
        '''Doing this from the clip paper, the loss function is the exact same '''

        v = F.normalize(model_vids, dim=1)
        t = F.normalize(model_text, dim=1)
        logits = v @t.T / 0.07

        targets = torch.arange(len(model_vids), device = model_vids.device)

        loss_v2t = F.cross_entropy(logits, targets)
        loss_t2v = F.cross_entropy(logits.T, targets)

        loss = (loss_v2t + loss_t2v) / 2
        return loss

