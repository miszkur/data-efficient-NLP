import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import os

from data_processing.ev_parser import create_dataloader
from tqdm.auto import tqdm
from torchmetrics import HammingDistance, F1Score
from visualisation.utils import plot_history
from sklearn.metrics import classification_report 

from transformers import get_cosine_schedule_with_warmup
from few_shot.data import FewShotDataLoader

class FewShotLearner:
  def __init__(self, device, model, config):
    self.num_tasks = 4
    self.hamming_distance = HammingDistance()
    self.num_classes = config.n_way
    self.f1_macro = F1Score(num_classes=self.num_classes, average='macro')
    self.device = device
    self.history = {
      'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []
      }
    self.bce_loss = nn.BCELoss()
    self.model = model
    if config.results_dir is not None:
      self.save_model_path = os.path.join(config.results_dir, 'bert' + '.pth')
    self.results_dir = config.results_dir

  def train(self, config, train_loader=None, validation_loader=None):
    self.n_way = config.n_way
    self.k_shot = config.k_shot
    num_epochs = config.num_epochs
    best_model = None
    epochs_val_loss_increase = 0
    self.max_grad_norm = config.max_grad_norm
    self.optimizer = optim.AdamW(
      self.model.parameters(),
      lr=config.lr,
      weight_decay=config.weight_decay)

    episode_loader = FewShotDataLoader()
    episode_valid_loader = FewShotDataLoader(split='valid')

    val_loss_prev = 1
    max_iter = 10
    for epoch_num in range(num_epochs):
      print('Epoch {}/{}'.format(epoch_num, num_epochs - 1))
      episode_iterator = tqdm(range(max_iter), total=max_iter)
      running_loss = 0.0
      accuracy = 0.0
      for iter in episode_iterator:
        loss, current_accuracy = self.training_step(episode_loader)

        accuracy += current_accuracy
        running_loss += loss
        episode_iterator.set_postfix(loss=(loss), accuracy=(current_accuracy))
      
      train_loss = running_loss / max_iter
      train_accuracy = accuracy / max_iter
      
      val_loss, val_accuracy, val_f1 = self.evaluate(episode_valid_loader)

      print(f'Training loss: {train_loss:.4f}, accuracy: {train_accuracy:.4f}')
      print(f'Validation Loss: {val_loss:.4f},  accuracy: {val_accuracy:.4f}, F1-score: {val_f1:.4f}')

      self.update_history_dict(train_loss, val_loss, train_accuracy, val_accuracy)

      if val_loss_prev < val_loss:
        epochs_val_loss_increase += 1
      else:
        best_model = self.model.state_dict()
        epochs_val_loss_increase = 0
      val_loss_prev = val_loss

      if self.should_stop_early(epochs_val_loss_increase):
        break

    self.model.load_state_dict(best_model)
    if self.results_dir is not None:
      torch.save(self.model.state_dict(), self.save_model_path)
      plot_history(history_dict=self.history, results_dir=self.results_dir)

  def inference(self, episode, return_cls=False):
    return self.model(episode)

  def training_step(self, episode_loader):

    self.model.train()
    self.optimizer.zero_grad()
    loss = None
    for i in range(self.num_tasks):
      episode = episode_loader.create_episode(
          n_classes=self.n_way, n_support=self.k_shot, n_query=self.k_shot)

      outputs = self.inference(episode)
      labels = torch.stack(episode['label']).to(self.device)

      if loss is None:
        loss = self.bce_loss(outputs, labels)
      else:
        loss += self.bce_loss(outputs, labels)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
      self.model.parameters(), self.max_grad_norm)
    self.optimizer.step()
    
    accuracy, _ = self.compute_metrics(outputs, labels)
    return loss.item(), accuracy

  def evaluate(self, episode_loader, classes=[]):
    self.model.eval() 
    acc_loss = 0
    acc_accuracy = 0
    acc_f1_score = 0
    per_class_results = {}
    for c in classes:
      per_class_results[c] = {'accuracy': 0, 'f1_score': 0}

    y_true = []
    y_pred = []
    eval_steps = 0
    with torch.no_grad():
      for i in range(25):
        episode, rand_keys = episode_loader.create_episode(
          n_classes=self.n_way, n_support=self.k_shot, n_query=self.k_shot, return_keys=True)
        outputs = self.inference(episode)
        labels = torch.stack(episode['label']).to(self.device)
        
        loss = self.bce_loss(outputs, labels)
        acc_loss += loss.item()
        print(outputs)
        print(rand_keys)
        print()
        accuracy, f1_score = self.compute_metrics(outputs, labels)
        acc_accuracy += accuracy
        acc_f1_score += f1_score

        y_true.append(labels.cpu().numpy())
        y_pred.append(np.round(torch.sigmoid(outputs).cpu().numpy()))
        eval_steps += 1

    acc_loss /= eval_steps
    acc_accuracy /= eval_steps
    acc_f1_score /= eval_steps

    if len(classes) == 0:
      return acc_loss, acc_accuracy, acc_f1_score

    # Per-class metrics.
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)  
    report = classification_report(
      y_true, y_pred, output_dict=True, zero_division=0)
    correctly_classified = (y_pred == y_true)
    incorrectly_classified = (y_pred != y_true)
    incorrectly_classified = incorrectly_classified.sum(axis=0)
    acc = correctly_classified.sum(axis=0) / y_pred.shape[0]
    per_class_results = {}
    for c in classes:
      per_class_results[c] = {
        'f1_score': report[f'{c}']['f1-score'], 
        'accuracy': acc[c],
        'precision': report[f'{c}']['precision'], 
        'recall': report[f'{c}']['recall'],
        'incorrect_predictions': incorrectly_classified[c]
      }
  
    # False positive + false negative
    num_predictions = len(y_true) * self.num_classes
    fp_fn = num_predictions - acc_accuracy * num_predictions

    return {
      'loss': acc_loss,
      'accuracy': acc_accuracy, 
      'f1_score': acc_f1_score, 
      'classes': per_class_results,
    }
    

  def compute_metrics(self, output, labels):
    """Compute accuracy and f1 score.
    
    Accuracy formula:
    accuracy = 1 - hamming distance
    """
    with torch.no_grad():
      preds = torch.sigmoid(output).cpu()
      target = labels.cpu().to(torch.int)
      acc = self.hamming_distance(preds, target)
      f1_score = self.f1_macro(preds, target)
      return 1 - acc.item(), f1_score.item()


  def update_history_dict(self, loss, val_loss, accuracy, val_accuracy):
    self.history['loss'].append(loss)
    self.history['accuracy'].append(accuracy)
    self.history['val_loss'].append(val_loss)
    self.history['val_accuracy'].append(val_accuracy)

  def should_stop_early(self, epochs_val_loss_increase):
    return (self.history['accuracy'][-1] > 0.98 and epochs_val_loss_increase > 1) \
            or epochs_val_loss_increase > 4

