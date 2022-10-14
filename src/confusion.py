import torch

EPS = 1e-7

# step functions and approximations
def indicator():
    ''' heaviside step function '''
    def f(x, t):
        return torch.where(x < t, 0, 1)
    return f

def sig(k=10):
    '''
        Kyurkchiev and Markov 2015
        Simple sigmoid
            - limits do not necessarily converge to the heaviside function
            - derivative can be 0
        = a/(1+e^(-k*(x-threshold)))
        for simplicity: a == 1
    '''
    def f(x, t):
        # shift to threshold
        x = x - t
        return 1/(1+torch.exp(-k*x))
    return f


def linear_approx(delta=0.2):
    ''' piecewise linear approximation of the Heaviside function
        x, t: pre-inverted (x, threshold) values in a tuple
        shape is 1 x num_thresholds
    '''
    d = delta
    def f(x,t):
        tt = torch.min(t, 1-t)
        cm1 = x < t - tt/2
        m1 = d/(t-tt/2)
        m2 = (1-2*d)/(tt+EPS)
        cm3 = x > t + tt/2
        m3 = d/(1-t-tt/2)
        res = torch.where(cm1, m1*x,
            torch.where(cm3, m3*x + (1-d-m3*(t+tt/2)),
                m2*(x-t)+0.5))
        return res
    return f

# helpers
def heaviside_sum(xs, thresholds, approx=None, gt_weight=None):
    ''' xs.shape: [batchsize, thresholds]
        thresholds.shape: [batchsize, thresholds]
        approx: linear_approx or approximation function to use
        '''
    a = approx(xs, thresholds).cuda()
    if gt_weight is not None:
        a = a * gt_weight.repeat(9,1).transpose(0,1)
    return torch.sum(a, axis=0)

# confusion matrix values
def l_tp(gt, pt, thresh, approx=None, gt_weight=None):
    # output closer to 1 if a true positive, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 1 -> (inverter = false)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (inverter = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (inverter = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = torch.where(thresh==0.0, torch.tensor([0.01], device=thresh.device),
            torch.where(thresh==1.0, torch.tensor([0.99], device=thresh.device), thresh))
    gt_t = torch.reshape(torch.repeat_interleave(gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = torch.reshape(torch.repeat_interleave(pt, thresh.shape[0]), (-1, thresh.shape[0]))
    condition = (pt_t > thresh)
    xs = torch.where(gt_t > 0, pt_t, torch.zeros_like(gt_t, dtype=pt_t.dtype))
    thresholds = torch.where(condition, thresh, 1-thresh)
    return heaviside_sum(xs, thresholds, approx, gt_weight=gt_weight)

def l_fn(gt, pt, thresh, approx=None, gt_weight=None):
    # output closer to 1 if a false negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 1 -> (inverter = true)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (inverter = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = torch.where(thresh==0.0, torch.tensor([0.01], device=thresh.device),
            torch.where(thresh==1.0, torch.tensor([0.99], device=thresh.device), thresh))
    gt_t = torch.reshape(torch.repeat_interleave(gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = torch.reshape(torch.repeat_interleave(pt, thresh.shape[0]), (-1, thresh.shape[0]))
    condition = (pt_t < thresh)
    xs = torch.where(gt_t > 0, 1-pt_t, torch.zeros_like(gt_t, dtype=pt_t.dtype))
    thresholds = torch.where(condition, 1-thresh, thresh)
    return heaviside_sum(xs, thresholds, approx, gt_weight=gt_weight)

def l_fp(gt, pt, thresh, approx=None, gt_weight=None):
    # output closer to 1 if a false positive, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (inverter = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 1 -> (inverter = false)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = torch.where(thresh==0.0, torch.tensor([0.01], device=thresh.device),
            torch.where(thresh==1.0, torch.tensor([0.99], device=thresh.device), thresh))
    gt_t = torch.reshape(torch.repeat_interleave(gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = torch.reshape(torch.repeat_interleave(pt, thresh.shape[0]), (-1, thresh.shape[0]))
    condition = (pt_t < thresh)
    xs = torch.where(gt_t > 0, torch.zeros_like(gt_t, dtype=pt_t.dtype), pt_t)
    thresholds = torch.where(condition, thresh, 1-thresh)
    return heaviside_sum(xs, thresholds, approx, gt_weight=gt_weight)

def l_tn(gt, pt, thresh, approx=None, gt_weight=None):
    # output closer to 1 if a true negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (invert = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (invert = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (invert = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 1 -> (invert = true)
    thresh = torch.where(thresh==0.0, torch.tensor([0.01], device=thresh.device),
            torch.where(thresh==1.0, torch.tensor([0.99], device=thresh.device), thresh))
    gt_t = torch.reshape(torch.repeat_interleave(gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = torch.reshape(torch.repeat_interleave(pt, thresh.shape[0]), (-1, thresh.shape[0]))
    condition = (pt_t < thresh)
    xs = torch.where(gt_t > 0, torch.zeros_like(gt_t, dtype=pt_t.dtype), 1-pt_t)
    thresholds = torch.where(condition, 1-thresh, thresh)
    return heaviside_sum(xs, thresholds, approx, gt_weight=gt_weight)

def confusion(gt, pt, thresholds, approx=None, class_weight=None):
    gt_weight = None
    if class_weight is not None:
        gt_weight = torch.where(gt==0, class_weight[0], class_weight[1])
    tp = l_tp(gt, pt, thresholds, approx, gt_weight=gt_weight)
    fn = l_fn(gt, pt, thresholds, approx, gt_weight=gt_weight)
    fp = l_fp(gt, pt, thresholds, approx, gt_weight=gt_weight)
    tn = l_tn(gt, pt, thresholds, approx, gt_weight=gt_weight)
    return tp, fn, fp, tn

# metrics
def bce(gt, pt):
    # (1/N) * ( p * ln(q) + (1-p) * ln(1-q) )
    return -1/gt.shape[0] * (gt * torch.log(pt) + (1-gt) * torch.log(1-pt)).nansum()

def fbeta(gt, pt, thresholds, approx=indicator(), beta=1):
    tp, fn, fp, tn = confusion(gt, pt, thresholds, approx)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    return (1 + beta*beta) * (precision * recall) / (beta*beta * precision + recall)

def accuracy(gt, pt, thresholds, approx=indicator()):
    tp, fn, fp, tn = confusion(gt, pt, thresholds, approx)
    return (tp + tn) / (tp + fn + fp + tn)

def kl(p, q):
    kl = p * torch.log(p/q)
    kl[torch.isnan(kl)] = 0
    kl[torch.isinf(kl)] = 0
    #print(f"kl: {kl}")
    return kl