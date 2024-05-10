from torchmetrics import Metric
import torch

# [TODO] Implement this!
# Calculate F1 score for whole val/test dataset not batch-wise
class MyF1Score(Metric):
    full_state_update: bool = True
    def __init__(self, cls_num):
        super().__init__()
        self.add_state('tp', default=torch.zeros(cls_num), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros(cls_num), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros(cls_num), dist_reduce_fx='sum')
        self.cls_num = cls_num
        
    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        for i in range(self.cls_num):
            tp = ((target==i) & (preds==i)).sum().item()
            fp = ((target!=i) & (preds==i)).sum().item()
            fn = ((target==i) & (preds!=i)).sum().item()
            self.tp[i] += tp
            self.fp[i] += fp
            self.fn[i] += fn
    
    def compute(self):
        precision = self.tp/(self.tp+self.fp)
        recall = self.tp/(self.tp+self.fn)
        f1_score = 2*(precision*recall)/(precision+recall)
        f1_score[torch.isnan(f1_score)] = 0
        return f1_score
    
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.numel() != target.numel():
            assert print("Prediction and target shapes are different")

        # [TODO] Count the number of correct prediction
        correct = (preds==target).sum().item()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()