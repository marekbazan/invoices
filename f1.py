import pandas as pd
import numpy as np
import cv2


def bb_intersection_over_union(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def calculateF1(detectionDf, annotationDf, results_path="results.csv"):
    """

    Args:
        detectionDf (pd.DataFrame): Wyjścia z modelu zapisane w pd.DataFrame(columns=['invoiceName', 'xmin', 'ymin', 'xmax', 'ymax'])
        (tak jak w pliku annotation_df).

        annotationDf (pd.DataFrame): Struktura pd.DataFrame(columns=['invoiceName', 'xmin', 'ymin', 'xmax', 'ymax'])
        z tagowaniem tabel wczytana z pliku annotation_df
        (
            import pandas as pd
            annotationDf = pd.read_csv('annotation_df.csv')
        )
        lub innego z takimi samymi nazwami kolumn.

        results_path (str, optional): Ścieżka do zapisu pliku CSV z wynikami. Defaults to "results.csv".
    """

    resultsDf = pd.DataFrame(
        columns=["IoU", "tablePrecision", "tableRecall", "tableF1"]
    )

    for IOUthreshold in np.arange(0.5, 1, 0.1).round(1):

        tableTP = 0
        tableFP = 0
        tableFN = 0

        for invoiceName, invoiceAnnotation in annotationDf.groupby("invoiceName"):

            invoiceAnnotation["detected"] = False

            detectedTables = detectionDf[detectionDf["invoiceName"] == invoiceName]

            for _, detectedBox in detectedTables.iterrows():

                detectedBox = detectedBox.loc["xmin":"ymax"].astype(int)

                invoiceAnnotation["intersection"] = invoiceAnnotation.apply(
                    lambda box: bb_intersection_over_union(
                        detectedBox, box["xmin":"ymax"]
                    ),
                    axis=1,
                )

                invoiceAnnotation["detected"] = invoiceAnnotation.apply(
                    lambda box: True
                    if (box["detected"] == False and box["intersection"] > 0)
                    else (True if box["detected"] == True else False),
                    axis=1,
                )

                if (invoiceAnnotation["intersection"] >= IOUthreshold).any():
                    tableTP += 1
                else:
                    tableFP += 1

            tableFN += (invoiceAnnotation["detected"] == False).sum()

        tablePrecision = (
            (tableTP) / (tableTP + tableFP) if (tableTP + tableFP) > 0 else 0
        )
        tablePrecision = round(tablePrecision, 3)

        tableRecall = (tableTP) / (tableTP + tableFN) if (tableTP + tableFN) > 0 else 0
        tableRecall = round(tableRecall, 3)

        tableF1 = (
            2 * (tablePrecision * tableRecall) / (tablePrecision + tableRecall)
            if (tablePrecision + tableRecall) > 0
            else 0
        )
        tableF1 = round(tableF1, 3)

        resultsDf.loc[len(resultsDf)] = [
            IOUthreshold,
            tablePrecision,
            tableRecall,
            tableF1,
        ]
        resultsDf.to_csv(results_path, index=None)
        print(
            f"IOU: {IOUthreshold}  |  Table  |  Precision: {tablePrecision}  |  Recall: {tableRecall}  |  F1: {tableF1}"
        )
        print()
