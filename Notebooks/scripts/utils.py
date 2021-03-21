from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import pandas as pd
import torch

class prediction:
    def __init__(self, ground_truth, top5_classes, top5_probs):
        self.ground_truth = ground_truth
        self.top5_classes = top5_classes
        self.top5_probs = top5_probs

    def get_gt(self):
        return self.ground_truth

    def get_top5_classes(self):
        return self.top5_classes

    def get_top5_probs(self):
        return self.top5_probs

def get_classes(probabilities, idx_to_class):
    # Most probable class
    top_probabilities, top_indices = probabilities.topk(5)

    top_probabilities = torch.nn.functional.softmax(top_probabilities, dim=1)

    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]

    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.
    #print(idx_to_class)

    top_classes = [idx_to_class[index] for index in top_indices]

    return top_probabilities, top_classes

def test_accuracy(model, test_loader):

    evaluation_results = []

    # Do validation on the test set
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():

        accuracy = 0

        for images, labels in iter(test_loader):

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            output = model.forward(images)

            probabilities = torch.exp(output)

            # Getting indices to their corresponding classes
            idx_to_class = {value: key for key, value in model.class_to_idx.items()}

            probs, classes = get_classes(probabilities, idx_to_class)

            # List with results to form a confusion matrix
            hr_label = labels.data.detach().type(torch.FloatTensor).numpy().tolist()[0]

            hr_label = idx_to_class[hr_label]

            pred = prediction(hr_label, classes, probs)
            evaluation_results.append(pred)

    print("Finished.")

    return evaluation_results

def results_pandas(model, test_loader):

    # Getting results
    results = test_accuracy(model, test_loader)

    gt = []
    top1 = []
    certainty = []

    results_dict = {'Ground truth': [],
                    'Top 1 prediction': [],
                    'Certainty': [],
                    'Top 1 Correct': [],
                    'Top 5 Correct': []}

    # Preparing data to create a Pandas DataFrame
    for result in results:
        results_dict['Ground truth'].append(result.get_gt())
        results_dict['Top 1 prediction'].append(result.get_top5_classes()[0])
        results_dict['Certainty'].append(result.get_top5_probs()[0])
        results_dict['Top 1 Correct'].append(1 if result.get_gt() == result.get_top5_classes()[0] else 0)
        results_dict['Top 5 Correct'].append(1 if result.get_gt() in result.get_top5_classes() else 0)

    results_df = pd.DataFrame(results_dict)

    return results_df

def get_scores(model, test_loader):
    results_df = results_pandas(model, test_loader)

    y_true = results_df['Ground truth'].tolist()
    y_pred = results_df['Top 1 prediction'].tolist()


    macro_f1 = f1_score(y_true = y_true,
                        y_pred = y_pred,
                        average = 'weighted',
                        zero_division=0)

    precision_s = precision_score(y_true = y_true,
                                  y_pred = y_pred,
                                  average = 'weighted',
                                  zero_division=0)

    recall_s = recall_score(y_true = y_true,
                            y_pred = y_pred,
                            average = 'weighted',
                            zero_division=0)

    top1_acc = results_df['Top 1 Correct'].mean()

    top5_acc = results_df['Top 5 Correct'].mean()

    return [top1_acc, top5_acc, precision_s, recall_s, macro_f1]
