class Evaluator():
    def __init__(self, data, metric_fn):
        self.data = data
        self.metric_fn = metric_fn


    def evaluate(self, model):
        return metric_fn(model, data, self.metric_fn)