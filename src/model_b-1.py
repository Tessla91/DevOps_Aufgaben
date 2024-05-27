from sklearn.model_selection import RepeatedStratifiedKFold

# ignore warning messages
# import warnings
# warnings.filterwarnings("ignore")

# evaluate a model
def evaluate_model(X, y, model):
# define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


