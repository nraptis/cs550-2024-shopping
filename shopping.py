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

    print("step?")
    print("labels = ", labels)
    print("evident = ", evidence)

    print("labels size = ", len(labels))
    for e in evidence:
        print("evi size: ", len(e))
        break

    print("Will try split")
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )
    print("Did then split")


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
        
    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as file:
        reader = csv.reader(file)
        first_read = True

        labels = []
        evidence = []
        for row in reader:
            print("row = ", row)
            if first_read:
                first_read = False
            else:

                list = []
                
                #0,
                #- Administrative, an integer
                list.append(int(row[0]))
               
                #0,
                #- Administrative_Duration, a floating point number
                list.append(float(row[1]))

                #0,
                #- Informational, an integer
                list.append(int(row[2]))

                #0,
                #- Informational_Duration, a floating point number
                list.append(float(row[3]))

                #1,
                #- ProductRelated, an integer
                list.append(int(row[4]))

                #0,
                #- ProductRelated_Duration, a floating point number
                list.append(float(row[5]))

                #0.2,
                #- BounceRates, a floating point number
                list.append(float(row[6]))
                
                #0.2,
                #- ExitRates, a floating point number
                list.append(float(row[7]))

                #0,
                #- PageValues, a floating point number
                list.append(float(row[8]))

                #0,
                #- SpecialDay, a floating point number
                list.append(float(row[9]))

                #Feb,
                #- Month, an index from 0 (January) to 11 (December)
                month_string = row[10].lower()

                if month_string == "jan":
                    list.append(0)
                elif month_string == "feb":
                    list.append(1)
                elif month_string == "mar":
                    list.append(2)
                elif month_string == "apr":
                    list.append(3)
                elif month_string == "may":
                    list.append(4)
                elif month_string == "june":
                    list.append(5)
                elif month_string == "jul":
                    list.append(6)
                elif month_string == "aug":
                    list.append(7)
                elif month_string == "sep":
                    list.append(8)
                elif month_string == "oct":
                    list.append(9)
                elif month_string == "nov":
                    list.append(10)
                else:
                    list.append(11)

                print("MON: ", month_string, "=>", list[-1])
                

                #1,
                #- OperatingSystems, an integer
                list.append(int(row[11]))

                #1,
                #- Browser, an integer
                list.append(int(row[12]))

                #1,
                #- Region, an integer
                list.append(int(row[13]))

                #1,
                #- TrafficType, an integer
                list.append(int(row[14]))
                
                #Returning_Visitor,
                #- VisitorType, an integer 0 (not returning) or 1 (returning)
                returning = row[15].lower()
                if returning == "returning_visitor":
                    list.append(1)
                else:
                    list.append(0)
                print("RV: ", returning)
                
                #FALSE,
                #- Weekend, an integer 0 (if false) or 1 (if true)
                weekend_str = row[16].strip().lower()
                weekend = 1 if weekend_str == "true" else 0
                
                #FALSE
                #Revenue
                revenue_str = row[17].strip().lower()
                label = 1 if revenue_str == "true" else 0

                evidence.append(list)
                labels.append(label)

    return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(evidence, labels)
    return classifier

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
    print("labels = ", len(labels))
    print("predictions = ", len(predictions))

    count = min(len(labels), len(predictions))

    labels_positive = 0
    labels_negative = 0

    predictions_true_positive = 0
    predictions_true_negative = 0
    
    for index in range(count):
        label = labels[index]
        prediction = predictions[index]
        if label == True:
            labels_positive += 1
            if prediction:
                predictions_true_positive += 1
        else:
            labels_negative += 1
            if not prediction:
                predictions_true_negative += 1
    
    sensitivity = 1.0
    specificity = 0.0

    if labels_positive > 0:
        sensitivity = predictions_true_positive / labels_positive
    if labels_negative > 0:
        specificity = predictions_true_negative / labels_negative

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
