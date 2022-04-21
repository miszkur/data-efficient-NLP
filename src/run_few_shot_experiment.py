import ml_collections
import config.config as configs
import torch
from few_shot.few_shot_learner import FewShotLearner
from few_shot.prototype_network import ProtoypeNetwork
import time

def run_supervised_experiment(
  config: ml_collections.ConfigDict, 
  device: str, 
  classes_to_track=[0,1]):

  # valid_loader, test_loader = create_dataloader(config, 'valid')
  results = {
    'accuracy': [], 
    'f1_score':[], 
    'train_time': []
  }
  # results['training_emissions'] = []

  for _ in range(5):
    model = ProtoypeNetwork(config=config.bert, device=device) 
    model.to(device)
    # Train
    learner = FewShotLearner(device, model, config.results_dir)
    # tracker = EmissionsTracker()
    # tracker.start()
    train_start_time = time.time()
    
    learner.train(config)

    results['train_time'].append(time.time() - train_start_time)
    # results['training_emissions'].append(tracker.stop())
    
    # Evaluate
    metrics = learner.evaluate(test_loader, classes=classes_to_track)
    loss = metrics['loss']
    accuracy = metrics['accuracy']
    f1_score = metrics['f1_score']
    print(f'Test loss: {loss}, accuracy: {accuracy}, f1 score: {f1_score}')
    results['accuracy'].append(accuracy)
    results['f1_score'].append(f1_score)
    for class_index in classes_to_track:
      for metric_name, value in metrics['classes'][class_index].items():
        if metric_name in list(results[class_index].keys()):
          results[class_index][metric_name].append(value)

    print('Saving results..')
    results_path = os.path.join(config.results_dir, f'SUPERVISED.pkl')
    with open(results_path, 'wb') as fp:
      pickle.dump(results, fp)


config = configs.multilabel_base()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

run_supervised_experiment(config, device)