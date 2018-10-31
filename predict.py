from argparse import ArgumentParser
from classifier import Classifier
from sklearn.metrics import classification_report


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument("--x_test_dir", type=str)
    parser.add_argument("--y_test_dir", type=str)
    parser.add_argument("--model_input_dir", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    opt = arg_parse()

    clf = Classifier()
    clf.load(opt.model_input_dir)
    images, labels = Classifier.loadData(opt.x_test_dir, opt.y_test_dir)

    X_test = Classifier.norm_and_bias(images)
    y_pred = clf.predict(X_test)
    y_true = list(labels)
    print(classification_report(y_true, y_pred))

