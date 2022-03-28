from transformers import BartForSequenceClassification, BartTokenizer
from torch import nn
from transformers import pipeline


class BartEntailment():
  def __init__(self, config):
    self.candidate_labels = config.class_names
    self.classifier = pipeline("zero-shot-classification",
                      model=config.bart)


  def predict(self, premise):
    return self.classifier(premise, self.candidate_labels, multi_label=True, device=1)
