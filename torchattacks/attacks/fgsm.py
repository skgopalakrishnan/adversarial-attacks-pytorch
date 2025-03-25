import torch
import torch.nn as nn

from ..attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, one_hot=True):
        super().__init__("FGSM", model)
        self.eps = eps
        self.supported_mode = ["default", "targeted"]
        self.one_hot = one_hot
        
    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        images.requires_grad = True
        outputs = self.get_logits(images)

        if not self.one_hot:
            loss = nn.CrossEntropyLoss()
            if self.targeted:      
                target_labels = self.get_target_label(images, labels)
        else:
            loss = nn.BCEWithLogitsLoss()
            # Convert labels (class indices) to one-hot vectors
            num_classes = outputs.shape[-1]
            if not self.targeted:
                labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float().to(self.device)
            else:
                target_labels = self.get_target_label(images, labels)
                target_labels = torch.nn.functional.one_hot(target_labels, num_classes=num_classes).float().to(self.device)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        adv_images = images + self.eps * grad.sign()
        # adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        # detach but no clamp
        adv_images = adv_images.detach() 

        return adv_images
