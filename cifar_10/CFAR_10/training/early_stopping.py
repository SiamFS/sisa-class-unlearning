import torch
import numpy as np
from datetime import datetime
import warnings

class SISAEarlyStopping:
    def __init__(self, 
                 patience=5,
                 min_delta=0.001,
                 restore_best_weights=True,
                 monitor='val_loss',
                 mode='min',
                 verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.early_stopped = False
        
        if self.verbose:
            print(f"Early stopping configured: patience={self.patience}, min_delta={self.min_delta}")
    
    def __call__(self, current_score, model=None, epoch=None):
        if self.mode == 'min': 
            score = -current_score
        else: 
            score = current_score
            
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch or 0
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score > self.best_score + self.min_delta:
            improvement = score - self.best_score
            self.best_score = score
            self.best_epoch = epoch or 0
            self.wait = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                direction = "decreased" if self.mode == 'min' else "increased"
                print(f"  {self.monitor} {direction} by {improvement:.6f} - saving best model")
        else:
            self.wait += 1
            if self.verbose and self.wait > 0:
                print(f"  No improvement for {self.wait}/{self.patience} epochs")
                
        if self.wait >= self.patience:
            self.stopped_epoch = epoch or 0
            self.early_stopped = True
            if self.verbose:
                print(f"   Early stopping triggered at epoch {self.stopped_epoch}")
                print(f"   Best {self.monitor}: {abs(self.best_score):.6f} at epoch {self.best_epoch}")
            return True
        return False
    
    def restore_best_model(self, model):
        if self.best_weights is not None and model is not None:
            model.load_state_dict({k: v.to(model.device if hasattr(model, 'device') else 'cpu') 
                                 for k, v in self.best_weights.items()})
            if self.verbose:
                print(f"   Restored best model from epoch {self.best_epoch}")
            return True
        return False

# Simple configuration function - just returns patience=5 always
def get_optimal_early_stopping_config(data_size=None, training_type='fresh'):
    """
    Simplified early stopping configuration - always returns patience=5
    """
    return {
        'patience': 5,
        'min_delta': 0.001,
        'monitor': 'val_loss',
        'mode': 'min',
        'restore_best_weights': True,
        'verbose': True
    }