import os
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import wandb
from warmup_scheduler import GradualWarmupScheduler

from src.utils.utils import transform_index_to_letter, calculate_levenshtein
from src.models.components.attention import plot_attention
from src.models.components.model_ema import ModelEMA


class Trainer:
    def __init__(self, cfg, model, device, verbose=True):
        self.cfg = cfg
        self.model = model.to(device)
        if cfg.trainer.model_ema:
            self.model_ema = ModelEMA(self.model)
        self.device = device
        self.verbose = verbose

        if cfg.trainer.criterion == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss(reduction='none')

        if cfg.trainer.optimizer.name.lower() == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(),
                                    lr=cfg.trainer.optimizer.lr,
                                    weight_decay=cfg.trainer.optimizer.weight_decay)
        elif cfg.trainer.optimizer.name.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=cfg.trainer.optimizer.lr,
                                   weight_decay=cfg.trainer.optimizer.weight_decay)
        elif cfg.trainer.optimizer.name.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=cfg.trainer.optimizer.lr,
                                  weight_decay=cfg.trainer.optimizer.weight_decay,
                                  momentum=cfg.trainer.optimizer.momentum)

        if cfg.trainer.scheduler.name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             'min',
                                                             factor=cfg.trainer.scheduler.lr_factor,
                                                             patience=cfg.trainer.scheduler.patience,
                                                             verbose=verbose)
        elif cfg.trainer.scheduler.name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                             T_max=(cfg.len_train_loader * cfg.epoch),
                                                             eta_min=cfg.trainer.scheduler.min_lr)
        else:
            scheduler = None

        if scheduler is not None and cfg.trainer.scheduler.warmup:
            scheduler = GradualWarmupScheduler(optimizer, 
                                               multiplier=1, 
                                               total_epoch=3*cfg.len_train_loader, 
                                               after_scheduler=scheduler)

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.start_epoch = 0
        if cfg.resume:
            checkpoint = torch.load(cfg.path.pretrained)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"Model loaded: {cfg.path.pretrained}")
        
        self.best_model = deepcopy(self.model)
        self.best_valid_dist = np.Inf
        self.es_patience = 0
        self.scaler = GradScaler() if cfg.mixed_precision else None
        
    def fit(self, train_loader, valid_loader):
        
        for epoch in range(self.start_epoch, self.start_epoch + self.cfg.epoch):
            train_loss = self.train_epoch(train_loader)
            print(f'\nEpoch: {epoch}')
            print(f'Train Loss: {train_loss:.6f}')
            valid_loss, valid_dist = self.validation(valid_loader)
            if self.cfg.trainer.scheduler.name == 'ReduceLROnPlateau':
                self.scheduler.step(valid_loss)
            print(f'Valid Loss: {valid_loss:.6f}, Valid dist: {valid_dist:.6f}')
            if not self.cfg.DEBUG:
                wandb.log({"train_loss": train_loss,
                           "valid_loss": valid_loss, 
                           "valid_dist": valid_dist,
                           "lr": self.optimizer.param_groups[0]['lr'],
                           "tr_rate": self.model.teacher_forcing_rate
                           })
        
            if valid_dist < self.best_valid_dist:
                if self.cfg.trainer.model_ema:
                    self.best_model = deepcopy(self.model_ema.ema)
                else:
                    self.best_model = deepcopy(self.model)
                self.best_valid_dist = valid_dist
                self.es_patience = 0
                if not self.cfg.DEBUG:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.best_model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'dist': valid_dist,
                        'loss': valid_loss,
                    }, os.path.join(self.cfg.path.weights, f'{self.cfg.name}-{self.cfg.dt_string}/{epoch}.pth')) 
                    print(f'Epoch {epoch} Model saved. ({self.cfg.name}-{self.cfg.dt_string}/{epoch}.pth)')
            elif epoch == (self.start_epoch + self.cfg.epoch - 1):
                if self.cfg.trainer.model_ema:
                    save_model = deepcopy(self.model_ema.ema)
                else:
                    save_model = deepcopy(self.model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': valid_loss,
                }, os.path.join(self.cfg.path.weights, f'{self.cfg.name}-{self.cfg.dt_string}/{epoch}_last.pth')) 
                print(f'Epoch {epoch} Model saved. ({self.cfg.name}-{self.cfg.dt_string}/{epoch}_last.pth)')
            else:
                self.es_patience += 1
                print(f"Valid dist. increased. Current early stop patience is {self.es_patience}")

            if (self.cfg.es_patience != 0) and (self.es_patience == self.cfg.es_patience):
                break

    def train_epoch(self, train_loader):
        self.model.train()

        losses = []
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, position=0, desc='Train')
        for batch_idx, (x, y, lx, ly) in pbar:

            self.optimizer.zero_grad()
            
            x = x.to(self.device)
            y = y.to(self.device)
        
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.cfg.mixed_precision:
                with autocast():
                    predictions, attentions = self.model(x, lx, y, mode='train', lr=current_lr)
                    
                    mask = torch.arange(y.shape[1]) < ly.unsqueeze(1)
                    mask = mask.to(self.device)
                    mask = mask.reshape(-1)

                    predictions = predictions.reshape(-1, predictions.shape[2])
                    y = y.reshape(-1)
                    loss = self.criterion(predictions, y)

                    masked_loss = torch.sum(loss * mask) / torch.sum(mask)

                losses.append(masked_loss.cpu().item())
                pbar.set_postfix(loss=masked_loss.cpu().item())
                self.scaler.scale(masked_loss).backward()
                if self.cfg.trainer.optimizer.clipping:
                    # Gradient Norm Clipping
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions, attentions = self.model(x, lx, y, mode='train', lr=current_lr)
                mask = torch.arange(y.shape[1]) < ly.unsqueeze(1)
                mask = mask.reshape(-1)
                mask = mask.to(self.device)

                predictions = predictions.reshape(-1, predictions.shape[2])
                y = y.reshape(-1)
                loss = self.criterion(predictions, y)

                masked_loss = torch.sum(loss*mask) / torch.sum(mask)

                losses.append(masked_loss.cpu().item())
                pbar.set_postfix(loss=masked_loss.cpu().item())
                masked_loss.backward()
                if self.cfg.trainer.optimizer.clipping:
                    # Gradient Norm Clipping
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
                self.optimizer.step()
            
            if self.cfg.trainer.model_ema:
                self.model_ema.update(self.model)

            if (self.scheduler is not None) and (self.cfg.trainer.scheduler.name != 'ReduceLROnPlateau'):
                self.scheduler.step()
            
            if self.cfg.DEBUG and batch_idx > 5:
                break

        return np.average(losses)

    def validation(self, valid_loader):
        if self.cfg.trainer.model_ema:
            model = self.model_ema.ema.eval()
        else:
            model = self.model.eval()

        losses, dists = [], []
        for batch_idx, (x, y, lx, ly) in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid', position=0, leave=True):
            
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                predictions, attentions = self.model(x, lx, y, mode='valid')
                
                dist = calculate_levenshtein(predictions, y)
                dists.append(dist)
                
                if not self.cfg.DEBUG and batch_idx == 0:
                    x_labels = [x for x in range(20)]
                    y_labels = [x for x in range(20)]
                    wandb.log({'attention map': wandb.plots.HeatMap(x_labels, y_labels, attentions[:20,:20], show_text=False)})
                
                mask = torch.arange(y.shape[1]) < ly.unsqueeze(1)
                mask = mask.reshape(-1)
                mask = mask.to(self.device)

                predictions = predictions.reshape(-1, predictions.shape[2])
                y = y.reshape(-1)
                loss = self.criterion(predictions, y)

                masked_loss = torch.sum(loss * mask) / torch.sum(mask)

                losses.append(masked_loss.cpu().item())

        torch.cuda.empty_cache()

        return np.average(losses), np.average(dist)

    def inference(self, test_loader):
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        letters = []
        for i, (x, lx) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Infer', position=0, leave=True):
            torch.cuda.empty_cache()
            
            x = x.to(self.device)

            with torch.no_grad():
                predictions, _ = self.best_model(x, lx, y=None, mode='eval')
                predictions = torch.argmax(predictions, dim=2).detach().cpu().numpy()

                letter = transform_index_to_letter(predictions)
                letters.extend(letter)

        return letters