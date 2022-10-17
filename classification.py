#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score


def print_conf_matrix(conf_matrix, title, ax):
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in conf_matrix.flatten()/np.sum(conf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='binary', ax=ax)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)

#%%
def analize_dataset(dataset, dataset_test = None):
    print('Analyzing dataset...')
    # ml training
    n_splits = 10
    max_iter = 2000
    n_estimators = 100
    random_state = 0

    # machine learning data
    if dataset_test is None:
        X = dataset.values[:,0:-1] # all columns except last one (income)
        Y = dataset.values[:,-1] # last column (income)

        test_size = 0.3

        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=test_size, random_state=random_state, shuffle=True)
    else:
        X_train = dataset.values[:,0:-1]
        Y_train = dataset.values[:,-1]
        X_validation = dataset_test.values[:,0:-1]
        Y_validation = dataset_test.values[:,-1]

    # ml evaluation
    models = []
    models.append(('LR', LogisticRegression(max_iter=max_iter)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('DT', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('RF', RandomForestClassifier(n_estimators=n_estimators)))
    models.append(('AB', AdaBoostClassifier(n_estimators=n_estimators)))
    models.append(('GB', GradientBoostingClassifier(n_estimators=n_estimators)))

    print('Computing models...')
    cv_results = []
    for name, model in models:
        kfold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        cv_score = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        cv_results.append(cv_score)
        print("Model %s: mean %f, std %f" % (name, cv_score.mean(), cv_score.std()))

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.boxplot(cv_results)
    ax.set_xticklabels([m[0] for m in models])
    ax.set_ylim([0.7, 1])
    plt.show()

    # ml classification
    print('Training models...')
    roc_curves = []
    metrics = []
    for i, (name, model) in enumerate(models):
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        
        accuracy = accuracy_score(Y_validation, predictions)
        precision = precision_score(Y_validation, predictions)
        recall = recall_score(Y_validation, predictions)
        f1 = f1_score(Y_validation, predictions)
        conf_matrix = confusion_matrix(Y_validation, predictions)

        metrics.append((name, accuracy, precision, recall, f1, conf_matrix))
        
        # print(classification_report(Y_validation, predictions))
        roc_metrics = roc_curve(Y_validation, predictions)
        roc_metrics_auc = roc_auc_score(Y_validation, predictions)
        roc_curves.append([roc_metrics, roc_metrics_auc])

        # if name == "LR":
        #     tf_model = tf.keras.models.Sequential()
        #     tf_model.add(tf.keras.Input(shape=(len(X_train[0]),)))
        #     tf_model.add(tf.keras.layers.Dense(1))

        #     # assign the parameters from sklearn to the TF model
        #     tf_model.layers[0].weights[0].assign(model.coef_.transpose())
        #     tf_model.layers[0].bias.assign(model.intercept_)

        #     # verify the models do the same prediction
        #     assert np.all((tf_model(X_validation) > 0)[:, 0].numpy() == model.predict(X_validation))

        #     inference_attack(tf_model, X_train, X_validation)
    
    # print combined utility metrics
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.2
    rects1 = ax.bar(x - 1.5 * width, [m[1] for m in metrics], width, label='accuracy')
    rects2 = ax.bar(x - 0.5 * width, [m[2] for m in metrics], width, label='precision')
    rects3 = ax.bar(x + 0.5 * width, [m[3] for m in metrics], width, label='recall')
    rects4 = ax.bar(x + 1.5 * width, [m[4] for m in metrics], width, label='f1')
    ax.bar_label(rects1, fmt='%.2f')
    ax.bar_label(rects2, fmt='%.2f')
    ax.bar_label(rects3, fmt='%.2f')
    ax.bar_label(rects4, fmt='%.2f')
    ax.set_title('Utility metrics')
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.legend(loc='lower left')
    ax.set_ylim([0, 1])
    fig.tight_layout()
    plt.show()

    # print combined confusion matrices
    fig, axs = plt.subplots(int(np.ceil(len(models) / 2)), 2, squeeze=False, figsize=(8, 10), dpi=70)
    for i, m in enumerate(metrics):
        print_conf_matrix(m[5], "Model %s" % m[0], axs[int(i / 2)][i % 2])
    plt.tight_layout()
    plt.show()

    # plot roc curves
    fig, ax = plt.subplots()
    for i, roc_metrics in enumerate(roc_curves):
        ax.plot(roc_metrics[0][0], roc_metrics[0][1], label="%s (auc = %0.2f)" % (models[i][0], roc_metrics[1]))
    ax.plot((0, 1), (0, 1), '--', color='Gray')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

#%%
# Just for doing some tests, it doesn't return any relevant information
# and is not used in the final code
def inference_attack(tf_model, X_train, X_validation):
    import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.membership_inference_attack as mia
    from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
    from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
    from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType

    print('Predict on train...')
    logits_train = tf_model.predict(X_train)
    print('Predict on test...')
    logits_test = tf_model.predict(X_validation)

    attack_input = AttackInputData(
        logits_train = logits_train,
        logits_test = logits_test,
        labels_train=None,
        labels_test=None
    )
    slicing_spec = SlicingSpec(
        entire_dataset = True,
        by_class = True,
        by_percentiles = False,
        by_classification_correctness = True
    )
    attack_types = [
        AttackType.THRESHOLD_ATTACK,
        AttackType.LOGISTIC_REGRESSION
    ] 
    attacks_result = mia.run_attacks(attack_input=attack_input, slicing_spec=slicing_spec, attack_types=attack_types)
    print(attacks_result.summary(by_slices=True))


#%%

# delta-DOCA algorithm
# applies the algorithm to the dataset with eps parameter
# saves it in a new file for caching purposes
# returns the resulting dataset
def deltaDoca(DATASET_PATH, DOCA_OUTPUT_PATH, eps):
    import os
    from pathlib import Path
    import DOCA_python
    if os.path.exists(DOCA_OUTPUT_PATH):
        doca_df = pd.read_csv(DOCA_OUTPUT_PATH)
    else:
        data = pd.read_csv(DATASET_PATH)
        last_column = data.iloc[:, -1]
        data = data.iloc[:, :-1] # delete the last column, it doesn't need to be anonymized
        std = np.std(data, axis=0)
        normalized_df: pd.DataFrame = data / std
        normalized_df: np.ndarray = normalized_df.to_numpy()

        doca_df = DOCA_python.doca(normalized_df, eps, beta=60)
        doca_df = pd.DataFrame(doca_df)
        doca_df.columns = data.columns # restore original column names
        doca_df = doca_df * std # revert normalization
        doca_df = pd.concat([doca_df, last_column], axis=1) # restore the last column
        
        filepath = Path(DOCA_OUTPUT_PATH)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        doca_df.to_csv(filepath, index=False)
    return doca_df

# %%

if __name__ == '__main__':
    DEFAULT_DATASET = "./datasets/adult_train.csv"
    DEFAULT_DATASET_TEST = "./datasets/adult_test.csv"
    DOCA_OUTPUT_BASE_PATH = "./tmp/"

    from argparse import ArgumentParser, BooleanOptionalAction

    parser = ArgumentParser(description='Run the analysis on the dataset')
    parser.add_argument("--dataset", dest="dataset_train", help="path to the dataset to use for training", default=DEFAULT_DATASET)
    parser.add_argument("--test", dest="dataset_test", help="path to the dataset to use for testing", default=DEFAULT_DATASET_TEST)
    parser.add_argument("--algorithm", dest="algorithm", help="anonymization algorithm (only \"doca\" is available)", default="doca")
    parser.add_argument("--eps", type=int, dest="eps", help="epsilon parameter for the doca algorithm", default=100)
    parser.add_argument("--skip-original", dest="skip_original", help="skip analysis of original dataset", default=False, action=BooleanOptionalAction)
    args = parser.parse_args()

    dataset_train = pd.read_csv(args.dataset_train)
    dataset_test = pd.read_csv(args.dataset_test)
    if not args.skip_original:
        analize_dataset(dataset_train, dataset_test)

    if args.algorithm == "doca":
        print('Applying delta-DOCA algorithm...')
        doca_ouput_file = args.dataset_train.split("/")[-1].replace(".csv", "_doca_" + str(args.eps) + ".csv")
        doca_df = deltaDoca(args.dataset_train, DOCA_OUTPUT_BASE_PATH + doca_ouput_file, args.eps)
        print('Analizing delta-DOCA dataset...')
        analize_dataset(doca_df, dataset_test)

# %%
