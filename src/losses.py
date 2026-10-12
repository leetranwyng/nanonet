import torch.nn.functional as F


def cakd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.3):
    """
    Compressed-Aware Knowledge Distillation (CAKD) loss.

    Combines cross-entropy with soft-target KL divergence distillation.
    The temperature T controls how soft the teacher's probability distribution
    is, and alpha controls the trade-off between the two terms.

    Args:
        student_logits: Raw logits from the student (or pruned/QAT) model.
        teacher_logits: Raw logits from the frozen teacher model.
        labels:         Ground-truth class indices (long tensor).
        T:              Distillation temperature (higher = softer targets).
        alpha:          Weight on the KD term (0 = pure CE, 1 = pure KD).

    Returns:
        Scalar combined loss.
    """
    ce_loss = F.cross_entropy(student_logits, labels)
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean',
    ) * (T ** 2)
    return (1 - alpha) * ce_loss + alpha * kd_loss
