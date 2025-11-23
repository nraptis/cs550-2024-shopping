import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    evidence = list()
    labels = []

    # Month lookup table (Month → Number 0–11)
    month_lut = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "June": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11
    }

    # Visitor Type lookup (string → 0 or 1)
    visitor_lut = {
        "Returning_Visitor": 1,
        "New_Visitor": 0,
        "Other": 0
    }

    # Boolean Type lookup (string → 0 or 1)
    boolean_lut = {
        "TRUE": 1,
        "FALSE": 0
    }

    with open(filename) as file:
        reader = csv.reader(file)

        # Skip the row of column names...
        next(reader)

        for row in reader:

            evidence_row = []
            label = 0

            # Administrative - Index 0, Integer
            evidence_row.append(int(row[0]))

            # Administrative_Duration - Index 1, Float
            evidence_row.append(float(row[1]))

            # Informational - Index 2, Integer
            evidence_row.append(int(row[2]))

            # Informational_Duration - Index 3, Float
            evidence_row.append(float(row[3]))

            # ProductRelated - Index 4, Integer
            evidence_row.append(int(row[4]))

            # ProductRelated_Duration - Index 5, Float
            evidence_row.append(float(row[5]))

            # BounceRates - Index 6, Float
            evidence_row.append(float(row[6]))

            # ExitRates - Index 7, Float
            evidence_row.append(float(row[7]))

            # PageValues - Index 8, Float
            evidence_row.append(float(row[8]))

            # SpecialDay - Index 9, Float
            evidence_row.append(float(row[9]))

            # Month - Index 10, String converted with month_lut
            month = month_lut[row[10]]
            evidence_row.append(month)

            # OperatingSystems - Index 11, Integer
            evidence_row.append(int(row[11]))

            # Browser - Index 12, Integer
            evidence_row.append(int(row[12]))

            # Region - Index 13, Integer
            evidence_row.append(int(row[13]))

            # TrafficType - Index 14, Integer
            evidence_row.append(int(row[14]))

            # VisitorType - Index 15, String converted with visitor_lut
            visitor = visitor_lut[row[15]]
            evidence_row.append(visitor)

            # Weekend - Index 16, "TRUE"/"FALSE" → Integer 0/1
            weekend = boolean_lut[row[16]]
            evidence_row.append(weekend)

            # Revenue - Index 17, "TRUE"/"FALSE" → label (0/1)
            label = boolean_lut[row[17]]

            evidence.append(evidence_row)
            labels.append(label)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    count = min(len(labels), len(predictions))

    actual_positive_count = sum(
        1 for i in range(count)
        if labels[i]
    )

    actual_negative_count = count - actual_positive_count

    true_positives = sum(
        1 for i in range(count)
        if labels[i] and predictions[i]
    )

    true_negatives = sum(
        1 for i in range(count)
        if not labels[i] and not predictions[i]
    )

    # Correctly predicted positives / actual number of positives.
    sensitivity = None
    if actual_positive_count:
        sensitivity = true_positives / actual_positive_count

    # Correctly predicted negatives / actual number of negatives.
    specificity = None
    if actual_negative_count:
        specificity = true_negatives / actual_negative_count

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
