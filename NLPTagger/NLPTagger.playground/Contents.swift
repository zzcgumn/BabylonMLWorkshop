import NaturalLanguage
import CreateML
import Foundation
import CoreML

let trainingData = try MLDataTable(contentsOf: Bundle.main.url(forResource: "BodyPartAndSymptom", withExtension: "json")!)

let tagger = try MLWordTagger(trainingData: trainingData, tokenColumn: "tokens", labelColumn: "labels")
let metadata = MLModelMetadata(author: "Martin Nygren",
                               shortDescription: "A custom NLP tagger to symptoms or body parts.",
                               license: "MIT",
                               version: "0.1")

let homePath = FileManager.default.homeDirectoryForCurrentUser
let desktopPath = homePath.appendingPathComponent("Desktop")
try tagger.write(to: desktopPath.appendingPathComponent("BodyPartAndSymptomTagger.mlmodel"), metadata: metadata)

let bodyPartAndSymptomScheme = NLTagScheme("Body part and symptom")
let bodyPartTag = NLTag("BODYPART")
let symptomTag = NLTag("SYMPTOM")

let modelURL = Bundle.main.url(forResource: "BodyPartAndSymptomTagger", withExtension: "mlmodelc")!
let bodyPartAndSymptomTaggerModel = try! NLModel(contentsOf: modelURL)

let bodyPartAndSymptomTagger = NLTagger(tagSchemes: [.nameType, bodyPartAndSymptomScheme])
bodyPartAndSymptomTagger.setModels([bodyPartAndSymptomTaggerModel], forTagScheme: bodyPartAndSymptomScheme)

let query = "I have a fever and my neck is stiff"
bodyPartAndSymptomTagger.string = query


bodyPartAndSymptomTagger.enumerateTags(in: query.startIndex..<query.endIndex,
                                       unit: .word,
                                       scheme: bodyPartAndSymptomScheme) {
                                        (tag, tokenRange) -> Bool in

                                        if tag == bodyPartTag {
                                            print("\(query[tokenRange]): BODYPART")
                                        }

                                        if tag == symptomTag {
                                            print("\(query[tokenRange]): SYMPTOM")
                                        }

                                        return true
}
