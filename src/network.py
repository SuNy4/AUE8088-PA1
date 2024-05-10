# Python packages
from termcolor import colored
from typing import Dict
import copy
import time

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
import torch.nn.functional as F
from torchvision import models, ops
from torchvision.models.alexnet import AlexNet
import torch

#wandb
import wandb

#Tensor RT
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

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
            nn.Conv2d(3, 64, kernel_size=9, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=1),
            #64x19x19
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #128x10x10

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            #256x4x4
        )
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        
        self.classifier=nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [TODO: Optional] Modify this as well if you want
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class layernorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute to B H W C for better calculation performance
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
    

class ConvNextBlock(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()

        self.block = nn.Sequential(
        nn.Conv2d(input_dim, input_dim, kernel_size=5, padding=2, groups=input_dim, bias=True),
        # permute for channel-wise linear calc.
        ops.Permute([0, 2, 3, 1]),
        nn.LayerNorm(input_dim),
        nn.Linear(in_features=input_dim, out_features=4 * input_dim, bias=True),
        nn.GELU(),
        #nn.Dropout(p=0.5),
        nn.Linear(in_features=4 * input_dim, out_features=input_dim, bias=True),
        ops.Permute([0, 3, 1, 2])
        )
        
        self.stochastic_depth = ops.StochasticDepth(0.1, "row")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


def layer(input_dim, output_dim, n_block):
    block = ConvNextBlock
    layer = []

    for _ in range(n_block):
        layer.append(block(input_dim))
    if output_dim is not None:
        layer.append(nn.Sequential(
            layernorm2d(input_dim),
            nn.Conv2d(input_dim, output_dim, kernel_size=2, stride=2)
        )
        )

    CNlayers = nn.Sequential(*layer)

    return CNlayers


class SOTAlike(AlexNet):
    def __init__(self, num_classes, last_channel = 768):
        super().__init__()
        norm_layer = layernorm2d

        self.base = ops.Conv2dNormActivation(
            in_channels=3,
            out_channels=96,
            kernel_size=4,
            stride=4,
            padding=0,
            norm_layer=norm_layer,
            activation_layer=None,
            bias=True,
            )
        
        self.features = nn.Sequential(
            layer(96, 192, 1),
            layer(192, 384, 1),
            layer(384, last_channel, 9),
            layer(last_channel, None, 1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            norm_layer(last_channel), 
            nn.Flatten(1), 
            nn.Linear(last_channel, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        x = self.features(x)
        x = self.avgpool(x)
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
        elif model_name == 'SOTAlike':
            self.model = SOTAlike(num_classes)
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        #Get config of hydra
        self.cfg = cfg

        # Metric
        self.accuracy = MyAccuracy()
        self.f1score = MyF1Score(cls_num=num_classes)

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(self.cfg)

    #Enter Params dict manually because of immutability
    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.type
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), lr=optim_params.lr)#, momentum=optim_params.momentum)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.type
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, gamma=scheduler_params.gamma)#, milestones=scheduler_params.milestones,)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y, _ = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_batch_start(self, batch, batch_idx):
        #Warm Up
        x, y = batch
        x = x.cuda()
        sample = torch.randn_like(x).cuda()
        for _ in range(10):
            self.forward(sample)

    def validation_step(self, batch, batch_idx):
        loss, scores, y, inf_t = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1score = self.f1score.update(scores, y)
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy, 'InferenceTime(ms)/Val': inf_t,},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = self.cfg.WANDB_IMG_LOG_FREQ)

    def on_validation_epoch_end(self):
        f1_score = self.f1score.compute()
        for i in range(len(f1_score)):
            self.log(f'Class{i:3d}', f1_score[i], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def _common_step(self, batch):
        x, y = batch

        #Check inference time
        start_t = time.time()  
        scores = self.forward(x)

        inf_t = (time.time()-start_t)*1000
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
            

class TRTclassifier(LightningModule):
    def __init__(self,
                num_classes: int = 200,
                file_path: str = 'file_path',
                optimizer_params: Dict = dict(),
                scheduler_params: Dict = dict(),
                ): 
        super().__init__()
        self.stream = None
        self.filepath = file_path
        self.num_classes = num_classes
        self.accuracy = MyAccuracy()
        self.f1score = MyF1Score(cls_num=num_classes)
    
    def load_RT_engine(self, filepath):
        # Loading Engine         
        file=open(filepath, "rb")
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        engine=runtime.deserialize_cuda_engine(file.read)
        self.context=engine.create_execution_context()
        
    def allocate_memory(self, batch):
        self.output = np.empty(self.num_classes, dtype=np.float32) 

        self.d_input = cuda.mem_alloc(1*batch.nbytes)
        self.d_output = cuda.mem_alloc(1*self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]

        self.stream = cuda.Stream()

    def validation_step(self, batch):
        x, y = batch
        self.allocate_memory(batch)
        # Async host to device, need to specify stream
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)

        t_start = time.time()
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        inf_t = time.time() - t_start

        # device to host memory
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)

        # Syncronize threads
        self.stream.synchronize()

        accuracy = self.accuracy(self.output, y)
        self.log_dict({'accuracy/val': accuracy, 'InferenceTime(ms)/Val': inf_t,},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
