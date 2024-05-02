from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('TP', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('FP', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('FN', default=torch.tensor(0), dist_reduce_fx='sum')
    
    def update(self, cls_num, preds, target):

        self.TP = torch.zeros(cls_num)
        self.FP = torch.zeros(cls_num)
        self.FN = torch.zeros(cls_num)
        preds = torch.argmax(preds, dim=1)

        for i in range(cls_num):
            tp = ((target==i) & (preds==i)).sum().item()
            fp = ((target!=i) & (preds==i)).sum().item()
            fn = ((target==i) & (preds!=i)).sum().item()
            self.TP[i] += tp
            self.FP[i] += fp
            self.FN[i] += fn
    
    def compute(self):
        precision = self.TP/(self.TP+self.FP)
        recall = self.TP/(self.TP+self.FN)
        f1_score = 2 * (precision * recall) / (precision + recall)
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
