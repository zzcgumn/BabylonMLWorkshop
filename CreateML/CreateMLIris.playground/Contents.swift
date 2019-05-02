import Foundation
import CreateML

let csvFile = Bundle.main.url(forResource: "IRIS", withExtension: "csv")!
let dataTable = try MLDataTable(contentsOf: csvFile)

print(dataTable)
let (evaluationTable, trainingTable) = dataTable.randomSplit(by: 0.20, seed: 5)

let classifier = try MLClassifier(trainingData: trainingTable, targetColumn: "species")

let trainingError = classifier.trainingMetrics.classificationError
let trainingAccuracy = (1.0 - trainingError) * 100

let validationError = classifier.validationMetrics.classificationError
let validationAccuracy = (1.0 - validationError) * 100

let classifierEvaluation = classifier.evaluation(on: evaluationTable)
let evaluationError = classifierEvaluation.classificationError
let evaluationAccuracy = (1.0 - evaluationError) * 100

let classifierMetadata = MLModelMetadata(author: "Vanndar Stormpike",
                                         shortDescription: "Predicts the Iris species.",
                                         version: "1.0")

let homePath = FileManager.default.homeDirectoryForCurrentUser
let desktopPath = homePath.appendingPathComponent("Desktop")
try classifier.write(to: desktopPath.appendingPathComponent("IRIS.mlmodel"), metadata: classifierMetadata)

