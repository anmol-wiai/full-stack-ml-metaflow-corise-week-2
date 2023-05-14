# TODO: In this cell, write your BaselineChallenge flow in the baseline_challenge.py file.

from metaflow import FlowSpec, step, Flow, current, Parameter, IncludeFile, card, current
from metaflow.cards import Table, Markdown, Artifact, Image
import numpy as np 
from dataclasses import dataclass

label_cutoff = 4
labeling_function = lambda row: 1 * (row["rating"] > label_cutoff) # TODO: Define your labeling function here.

@dataclass
class ModelResult:
    "A custom struct for storing model evaluation results."
    name: None
    params: None
    pathspec: None
    acc: None
    rocauc: None

class BaselineChallenge(FlowSpec):

    split_size = Parameter('split-sz', default=0.2)
    data = IncludeFile('data', default='/home/workspace/workspaces/full-stack-ml-metaflow-corise-week-1/data/Womens Clothing E-Commerce Reviews.csv')
    kfold = Parameter('k', default=5)
    scoring = Parameter('scoring', default='accuracy')

    @step
    def start(self):

        import pandas as pd
        import io 
        from sklearn.model_selection import train_test_split
        
        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))
        # TODO: load the data. 
        # Look up a few lines to the IncludeFile('data', default='Womens Clothing E-Commerce Reviews.csv'). 
        # You can find documentation on IncludeFile here: https://docs.metaflow.org/scaling/data#data-in-local-files

        # filter down to reviews and labels 
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df = df[~df.review_text.isna()]
        df['review'] = df['review_text'].astype('str')
        _has_review_df = df[df['review_text'] != 'nan']
        reviews = _has_review_df['review_text']
        labels = _has_review_df.apply(labeling_function, axis=1)
        self.df = pd.DataFrame({'label': labels, **_has_review_df})

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({'review': reviews, 'label': labels})
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f'num of rows in train set: {self.traindf.shape[0]}')
        print(f'num of rows in validation set: {self.valdf.shape[0]}')

        self.next(self.baseline, self.model)


    @step
    def baseline(self):
        "Compute the baseline"

        from sklearn.metrics import accuracy_score, roc_auc_score
        self._name = "baseline"
        params = "Always predict the majority class: "
        pathspec = f"{current.flow_name}/{current.run_id}/{current.step_name}/{current.task_id}"
        
        majority_class = self.traindf["label"].mode()[0]
        params += str(majority_class)

        val_pred = [majority_class] * len(self.valdf)
        val_score = [majority_class] * len(self.valdf)
        predictions = val_pred # TODO: predict the majority class
        acc = accuracy_score(self.valdf["label"], val_pred) # TODO: return the accuracy_score of these predictions
         
        rocauc = roc_auc_score(self.valdf["label"], val_score) # TODO: return the roc_auc_score of these predictions
        self.result = ModelResult("Baseline", params, pathspec, acc, rocauc)
        self.next(self.aggregate)


    @step
    def model(self):

        # TODO: import your model if it is defined in another file.
        from model import NbowModel

        self._name = "model"
        # NOTE: If you followed the link above to find a custom model implementation, 
            # you will have noticed your model's vocab_sz hyperparameter.
            # Too big of vocab_sz causes an error. Can you explain why? 
            # > Because of linear layer. Linear layer assumes `vocab_sz` number of input features. 
            # > If we set `vocab_sz` to a very large value, CountVectorizer doesn't see that many unique words, 
            # > and returns lesser number of features. 
            # > (Like linear layer expects 1 million features but CountVectorizer gives only 800 features!)
        self.model_params = [{'vocab_sz': 100}, {'vocab_sz': 300}, {'vocab_sz': 500}, {'vocab_sz': 700}]
        self.fit_params = [{"batch_size": 32, "epochs": 5}, {"batch_size": 32, "epochs": 10}, {"batch_size": 32, "epochs": 20}, 
                           {"batch_size": 256, "epochs": 5}, {"batch_size": 256, "epochs": 10}, {"batch_size": 256, "epochs": 20},
                           {"batch_size": 512, "epochs": 5}, {"batch_size": 512, "epochs": 10}, {"batch_size": 512, "epochs": 20},]

        pathspec = f"{current.flow_name}/{current.run_id}/{current.step_name}/{current.task_id}"

        self.results = []
        for model_params in self.model_params:
            for fit_params in self.fit_params:
                params = {**model_params, **fit_params}

                model = NbowModel(**model_params) # TODO: instantiate your custom model here!
                model.fit(X=self.traindf['review'].to_numpy(), y=self.traindf['label'].to_numpy(), **fit_params)
                acc = model.eval_acc(X=self.valdf["review"].to_numpy(), labels=self.valdf["label"].to_numpy()) # TODO: evaluate your custom model in an equivalent way to accuracy_score.
                rocauc = model.eval_rocauc(X=self.valdf["review"].to_numpy(), labels=self.valdf["label"].to_numpy()) # TODO: evaluate your custom model in an equivalent way to roc_auc_score.

                self.results.append(ModelResult(f"NbowModel (vocab, batch, epoch) - {list(params.values())}", params, pathspec, acc, rocauc))

        self.next(self.aggregate)


    def add_one(self, rows, result, df):
        "A helper function to load results."
        rows.append([
            Markdown(result.name),
            Artifact(result.params),
            Artifact(result.pathspec),
            Artifact(result.acc),
            Artifact(result.rocauc)
        ])
        df['name'].append(result.name)
        df['accuracy'].append(result.acc)
        return rows, df

    @card(type="corise") # TODO: Set your card type to "corise". 
            # I wonder what other card types there are?
            # https://docs.metaflow.org/metaflow/visualizing-results
            # https://github.com/outerbounds/metaflow-card-altair/blob/main/altairflow.py
    @step
    def aggregate(self, inputs):
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib import rcParams 
        rcParams.update({'figure.autolayout': True})

        rows = []
        plot_df = {'name': [], 'accuracy': []}
        for task in inputs:
            if task._name == "model": 
                for result in task.results:
                    print(result)
                    rows, plot_df = self.add_one(rows, result, plot_df)
            elif task._name == "baseline":
                print(task.result)
                rows, plot_df = self.add_one(rows, task.result, plot_df)
            else:
                raise ValueError("Unknown task._name type. Cannot parse results.")
            
        current.card.append(Markdown("# All models from this flow run"))

        # TODO: Add a Table of the results to your card! 
        current.card.append(
            Table(
                rows, # TODO: What goes here to populate the Table in the card? 
                headers=["Model name", "Params", "Task pathspec", "Accuracy", "ROCAUC"]
            )
        )
        
        fig, ax = plt.subplots(1,1, dpi=300)
        plt.xticks(rotation=90)
        sns.barplot(data=pd.DataFrame(plot_df), x="name", y="accuracy", ax=ax)
        
        # TODO: Append the matplotlib fig to the card
        # Docs: https://docs.metaflow.org/metaflow/visualizing-results/easy-custom-reports-with-card-components#showing-plots
        current.card.append(Image.from_matplotlib(fig))

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    BaselineChallenge()
