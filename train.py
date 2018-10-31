from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from classifier import Classifier


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument("--x_train_dir", type=str)
    parser.add_argument("--y_train_dir", type=str)
    parser.add_argument("--model_output_dir", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    opt = arg_parse()

    images, labels = Classifier.loadData(opt.x_train_dir, opt.y_train_dir)
    Y = Classifier.prepare_y_ohe(labels)
    X = Classifier.norm_and_bias(images)

    clf = Classifier()
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf.fit(X_train, Y_train, X_validate, Y_validate)

    clf.dump(opt.model_output_dir)

    y_pred = clf.predict(X)
    y_true = list(labels)
    print(classification_report(y_true, y_pred))