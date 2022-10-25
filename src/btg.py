import torch
from confusion import confusion, sig, linear_approx, fbeta, kl, bce, EPS

def mean_fbeta_approx_loss_on(device, thresholds=torch.arange(0.1, 1, 0.1), beta=1, approx=linear_approx(), class_weight=None):
    print(f"mean_fbeta_approx_loss_on beta: '{beta}'")
    thresholds = thresholds.to(device)
    def loss(pt, gt):
        """Approximate F1:
            - Linear interpolated Heaviside function
            - Harmonic mean of precision and recall
            - Mean over a range of thresholds
        """
        classes = pt.shape[1] if len(pt.shape) == 2 else 1
        mean_f1s = torch.zeros(classes, dtype=torch.float32).to(device)
        # mean over all classes
        for i in range(classes):
            thresholds = torch.arange(0.1, 1, 0.1).to(device)
            tp, fn, fp, tn = confusion(gt, pt[:,i] if classes > 1 else pt, thresholds, approx, class_weight=class_weight)
            precision = tp/(tp+fp+EPS)
            recall = tp/(tp+fn+EPS)
            f1 = (1+beta*beta) * (precision * recall) / (beta*beta * precision + recall + EPS)
            mean_f1s[i] = torch.mean(f1)
        loss = 1 - mean_f1s.mean()
        return loss, tp, fn, fp, tn
    return loss

def mean_accuracy_approx_loss_on(device, thresholds=torch.arange(0.1, 1, 0.1), approx=linear_approx(), class_weight=None):
    thresholds = thresholds.to(device)
    def loss(pt, gt):
        """Approximate Accuracy:
            - Linear interpolated Heaviside function
            - (TP + TN) / (TP + TN + FP + FN)
            - Mean over a range of thresholds
        """
        classes = pt.shape[1] if len(pt.shape) == 2 else 1
        mean_accs = torch.zeros(classes, dtype=torch.float32).to(device)
        # mean over all classes
        for i in range(classes):
            tp, fn, fp, tn = confusion(gt, pt[:,i] if classes > 1 else pt, thresholds, approx, class_weight=class_weight)
            acc = (tp + tn) / (tp + tn + fp + fn)
            mean_accs[i] = torch.mean(acc)
        loss = 1 - mean_accs.mean()
        return loss, tp, fn, fp, tn
    return loss

def area(x,y):
    ''' area under curve via trapezoidal rule '''
    xs, idxs = torch.sort(x)
    ys = y[idxs]
    return torch.trapz(ys, xs)

def mean_auroc_approx_loss_on(device, thresholds=torch.arange(0.1, 1, 0.1), approx=linear_approx(), class_weight=None):
    thresholds = thresholds.to(device)
    def loss(pt, gt):
        """Approximate auroc:
            - Linear interpolated Heaviside function
            - roc (11-point approximation)
            - integrate via trapezoidal rule under curve
        """
        classes = pt.shape[1] if len(pt.shape) == 2 else 1
        areas = []
        # mean over all classes
        for i in range(classes):
            tp, fn, fp, tn = confusion(gt, pt[:,i] if classes > 1 else pt, thresholds, approx, class_weight=class_weight)
            fpr = fp/(fp+tn+EPS)
            tpr = tp/(tp+fn+EPS)
            a = area(fpr, tpr)
            if a > 0:
                areas.append(a)

        if gt.sum() < 1 and len(areas) < 1:
            areas.append(torch.tensor(0.0).to(device))
        loss = 1 - torch.stack(areas).mean()
        return loss, tp, fn, fp, tn
    return loss
