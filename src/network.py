# Python packages
from termcolor import colored
from typing import Dict
import copy
import time

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch

#wandb
import wandb

#Tensor RT
import tensorrt as trt
import pycuda.driver as cuda

# Custom packages
from src.metric import MyAccuracy, MyF1Score
#import src.config as cfg
from src.util import show_setting

# [TODO: Optional] Rewrite this class if you want
class MyNetwork(AlexNet):
    def __init__(self, num_classes):
        super().__init__()
        dropout = 0.5
        # [TODO] Modify feature extractor part in AlexNet
        #input img = 3x64x64
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            #64x32x32
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            #128x16x16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            #256x8x8

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            #512x8x8

            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            )
        
        # self.avgpool=nn.Sequential(

        #     nn.AdaptiveAvgPool2d((6,6)),
        #     #256x6x6
        #     #nn.Dropout(p=dropout),
        #     )
        
        # self.classifier=nn.Sequential(
        #     nn.Linear(256*6*6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [TODO: Optional] Modify this as well if you want
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SimpleClassifier(LightningModule):
    def __init__(self, cfg,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork(num_classes)
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        #Get config of hydra
        self.cfg = cfg

        #Blank list for calculating F1 score at validation epoch end
        self.valid_output = []
        self.target = []

        # Metric
        self.accuracy = MyAccuracy()
        #self.f1 = F1Score(task='multiclass', num_classes=num_classes)
        self.f1score = MyF1Score(cls_num=num_classes)

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(self.cfg)

    #Enter Params dict manually because of immutability
    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.type
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), lr=optim_params.lr, momentum=optim_params.momentum)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.type
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, milestones=scheduler_params.milestones, gamma=scheduler_params.gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        mode = "Train"
        loss, scores, y, _ = self._common_step(batch, mode)
        accuracy = self.accuracy(scores, y)
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mode = "Val"
        loss, scores, y, inf_t = self._common_step(batch, mode)
        accuracy = self.accuracy(scores, y)
        self.valid_output.append(scores)
        self.target.append(y)
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy, 'InferenceTime(ms)/Val': inf_t,},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = self.cfg.WANDB_IMG_LOG_FREQ)

    def on_validation_epoch_end(self):
        preds = torch.cat(self.valid_output, dim=0)
        gt = torch.cat(self.target, dim=0)
        self.wandb_log_f1socre(self.cfg.NUM_CLASSES, preds, gt)
        # Clear valid, target variable for next epoch
        self.valid_output=[]
        self.target=[]

    def _common_step(self, batch, mode):
        x, y = batch

        # Warm up
        if mode == "Val":
            x = x.cuda()
            sample = torch.randn_like(x).cuda()
            for _ in range(10):
                self.forward(sample)
        
        #Check inference time
        start_t = time.time()  
        scores = self.forward(x)
        fin_t = time.time()

        inf_t = (fin_t-start_t)*1000
        loss = self.loss_fn(scores, y)
        return loss, scores, y, inf_t

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
            
    def wandb_log_f1socre(self, cls_num, scores, y):
        f1_score = self.f1score(cls_num, scores, y)
        for i in range(len(f1_score)):
            self.log(f'Class{i:3d}', f1_score[i], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def load_RT_engine(filepath):           
        file = open(filepath, "rb")
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        engine = runtime.deserialize_cuda_engine(file.read)
        
    def allocate_memory(self, batch):
        self.output = np.empty(self.num_classes, dtype=self.target_dtype) # Need to set both input and output precisions to FP16 to fully enable FP16

        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]

        self.stream = cuda.Stream()

    def predict(self, batch, eval_exec_time = False): # result gets copied into output
        if self.stream is None:
            self.allocate_memory(batch)

        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)

        # Execute model
        if eval_exec_time:
            t_start = time.time()
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        if eval_exec_time:
            t_inference = time.time() - t_start
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()

        return (t_inference, self.output) if eval_exec_time else self.output
