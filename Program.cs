using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Text_Classification_ML
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            //Console.WriteLine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName);

            List<string> FilesToClassify = new List<string>()
                                    {
                                        "Dennis+Schwartz",
                                        "James+Berardinelli",
                                        //"Scott+Renshaw",
                                        //"Steve+Rhodes",
                                    };

            List<string> FilesToTest = new List<string>()
                                    {
                                        //"Dennis+Schwartz",
                                        //"James+Berardinelli",
                                        "Scott+Renshaw",
                                        //"Steve+Rhodes",
                                    };

            List<string> FilesToValidate = new List<string>()
                                    {
                                        //"Dennis+Schwartz",
                                        //"James+Berardinelli",
                                        //"Scott+Renshaw",
                                        "Steve+Rhodes",
                                    };

            List<TextML> TextsToTrain = new List<TextML>();
            foreach (var name in FilesToClassify)
                TextsToTrain.Add(GetFile(name));

            List<TextML> TextsToTest = new List<TextML>();
            foreach (var name in FilesToTest)
                TextsToTest.Add(GetFile(name));

            List<TextML> TextsToValidate = new List<TextML>();
            foreach (var name in FilesToValidate)
                TextsToValidate.Add(GetFile(name));

            List<String> Files1 = new List<string>() { FilesToClassify.First() };
            List<String> Files2 = new List<string>() { FilesToClassify.Last() };

            //Naive Baise model
            var mlContextNB = new MLContext();

            var trainDataNB = LoadDataNB(mlContextNB, TextsToTrain, FilesToClassify);
            var testDataNB = LoadDataNB(mlContextNB, TextsToTest, FilesToTest);
            var validateDataNB = LoadDataNB(mlContextNB, TextsToValidate, FilesToValidate);
            var modelNB = BuildAndTrainModelNB(mlContextNB, trainDataNB);
            EvaluateNB(mlContextNB, modelNB, testDataNB);
            ValidateNB(mlContextNB, validateDataNB);


            //SVM model
            var mlContextSVM = new MLContext();

            var trainDataSVM = LoadDataSVM(mlContextSVM, TextsToTrain, FilesToClassify);
            var testDataSVM = LoadDataSVM(mlContextSVM, TextsToTest, FilesToTest);
            var validateDataSVM = LoadDataSVM(mlContextSVM, TextsToValidate, FilesToValidate);
            var modelSVM = BuildAndTrainModelSVM(mlContextSVM, trainDataSVM);
            EvaluateSVM(mlContextSVM, modelSVM, testDataSVM);
            ValidateSVM(mlContextSVM, validateDataSVM);



            #region TestModels
            /*

                        int exampleB = 0;
                        int exampleG = 1300;
                        int exampleN = 650;
                        var sampleStatementB = new TextData() { TextSubj = Texts.Last().TextLinesArr[exampleB] };
                        var sampleStatementG = new TextData() { TextSubj = Texts.Last().TextLinesArr[exampleG] };
                        var sampleStatementN = new TextData() { TextSubj = Texts.Last().TextLinesArr[exampleN] };
                        *//*            var sampleStatement1 = new TextData() { TextSubj = "Bad Bad Bad Bad Bad Bad Bad Bad Bad Bad Bad Bad Bad Bad Bad Bad Bad " };
                                    var sampleStatement2 = new TextData() { TextSubj = " The best  The best  The best  The best  The best " };
                                    var sampleStatement3 = new TextData() { TextSubj = "Neutral Neutral Neutral Neutral Neutral Neutral Neutral Neutral Neutral " };
                                    var resultPredictionBad = predictionFunction.Predict(sampleStatement1);
                                    var resultPredictionGood = predictionFunction.Predict(sampleStatement2);
                                    var resultPredictionNeutral = predictionFunction.Predict(sampleStatement3);
                        *//*
                        var resultPredictionB = predictionFunction.Predict(sampleStatementB);
                        var resultPredictionG = predictionFunction.Predict(sampleStatementG);
                        var resultPredictionN = predictionFunction.Predict(sampleStatementN);
                        Console.WriteLine();
                        Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

                        Console.WriteLine();
                        Console.WriteLine($"Rating: {Texts.Last().ReviewMLs.ElementAt(exampleB).Rating} | Prediction: {(Convert.ToBoolean(resultPredictionB.Prediction) ? "Negative" : "Positive")} | Probability: {resultPredictionB.Probability} ");
                        Console.WriteLine($"Rating: {Texts.Last().ReviewMLs.ElementAt(exampleG).Rating} | Prediction: {(Convert.ToBoolean(resultPredictionG.Prediction) ? "Negative" : "Positive")} | Probability: {resultPredictionG.Probability} ");
                        Console.WriteLine($"Rating: {Texts.Last().ReviewMLs.ElementAt(exampleN).Rating} | Prediction: {(Convert.ToBoolean(resultPredictionN.Prediction) ? "Negative" : "Positive")} | Probability: {resultPredictionN.Probability} ");
                        //Console.WriteLine($"Review: {resultPredictionBad.TextSubj} | Prediction: {(Convert.ToBoolean(resultPredictionBad.Prediction) ? "Negative" : "Positive")} | Probability: {resultPredictionBad.Probability} ");
                        //Console.WriteLine($"Review: {resultPredictionGood.TextSubj} | Prediction: {(Convert.ToBoolean(resultPredictionGood.Prediction) ? "Negative" : "Positive")} | Probability: {resultPredictionGood.Probability} ");
                        //Console.WriteLine($"Review: {resultPredictionNeutral.TextSubj} | Prediction: {(Convert.ToBoolean(resultPredictionNeutral.Prediction) ? "Negative" : "Positive")} | Probability: {resultPredictionNeutral.Probability} ");

                        Console.WriteLine("=============== End of Predictions ===============");
                        Console.WriteLine();
                        //var label3class = mlContext.Model.CreatePredictionEngine<TextData, TextPrediction>(model).Predict(review);
                    */
            #endregion
            Console.ReadKey();
        }

        private static IDataView LoadDataNB(MLContext mlContext, List<TextML> listTexts, List<string> files)
        {
            List<TextDataNB> textData = new List<TextDataNB>();

            foreach (var file in files)
            {
                foreach (var review in listTexts.Where(t => t.Name == file).ToList().First().ReviewMLs)
                {
                    textData.Add(new TextDataNB() { Label4Class = review.Label4Class, Rating = review.Rating, TextIsToxic = review.IsToxic, Label3Class = review.Label3Class, TextSubj = review.Subj });
                }
            }

            IDataView trainingData = mlContext.Data.LoadFromEnumerable(textData);
            return trainingData;
        }

        private static IDataView LoadDataSVM(MLContext mlContext, List<TextML> listTexts, List<string> files)
        {
            List<TextDataSVM> textData = new List<TextDataSVM>();

            foreach (var file in files)
            {
                foreach (var review in listTexts.Where(t => t.Name == file).ToList().First().ReviewMLs)
                {
                    textData.Add(new TextDataSVM() { Label4Class = review.Label4Class, Rating = review.Rating, TextIsToxic = review.IsToxic, Label3Class = review.Label3Class, TextSubj = review.Subj });
                }
            }

            IDataView trainingData = mlContext.Data.LoadFromEnumerable(textData);
            return trainingData;
        }

        private static IDataView LoadDataValidateNB(MLContext mlContext, List<TextML> listTexts, List<string> files)
        {
            List<TextDataNB> textData = new List<TextDataNB>();

            foreach (var file in files)
            {
                foreach (var review in listTexts.Where(t => t.Name == file).ToList().First().ReviewMLs)
                {
                    textData.Add(new TextDataNB() { Label4Class = review.Label4Class, Rating = review.Rating, TextIsToxic = review.IsToxic, Label3Class = review.Label3Class, TextSubj = review.Subj });
                }
            }

            var data = mlContext.Data.LoadFromEnumerable(textData);

            // Define data prep estimator
            IEstimator<ITransformer> dataPrepEstimator =
                mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "Features")
                         .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", new string[] { "Features" }))
                         .Append(mlContext.Transforms.NormalizeMinMax("Features"));

            // Create data prep transformer
            ITransformer dataPrepTransformer = dataPrepEstimator.Fit(data);

            // Transform data
            IDataView transformedData = dataPrepTransformer.Transform(data);
            return transformedData;
        }

        private static IDataView LoadDataValidateSVM(MLContext mlContext, List<TextML> listTexts, List<string> files)
        {
            List<TextDataSVM> textData = new List<TextDataSVM>();

            foreach (var file in files)
            {
                foreach (var review in listTexts.Where(t => t.Name == file).ToList().First().ReviewMLs)
                {
                    textData.Add(new TextDataSVM() { Label4Class = review.Label4Class, Rating = review.Rating, TextIsToxic = review.IsToxic, Label3Class = review.Label3Class, TextSubj = review.Subj });
                }
            }

            var data = mlContext.Data.LoadFromEnumerable(textData);

            // Define data prep estimator
            IEstimator<ITransformer> dataPrepEstimator =
                mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "Features")
                         .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", new string[] { "Features" }))
                         .Append(mlContext.Transforms.NormalizeMinMax("Features"));

            // Create data prep transformer
            ITransformer dataPrepTransformer = dataPrepEstimator.Fit(data);

            // Transform data
            IDataView transformedData = dataPrepTransformer.Transform(data);
            return transformedData;
        }

        public static ITransformer BuildAndTrainModelNB(MLContext mlContext, IDataView trainingData)
        {
            var estimator = mlContext.Transforms
                          .Conversion.MapValueToKey("Label")
                          .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "Features"))
                          .Append(mlContext.MulticlassClassification.Trainers.NaiveBayes(labelColumnName: "Label", featureColumnName: "Features"));

            Console.WriteLine("=============== Create and Train the NB-Model ===============");
            var model = estimator.Fit(trainingData);
            Console.WriteLine("=============== End of training NB ===============");
            Console.WriteLine();

            return model;
        }

        public static ITransformer BuildAndTrainModelSVM(MLContext mlContext, IDataView trainingData)
        {
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "Features")
                          .Append(mlContext.BinaryClassification.Trainers.LinearSvm("Label", featureColumnName: "Features"));

            Console.WriteLine("=============== Create and Train the SVM-Model ===============");
            var model = estimator.Fit(trainingData);
            Console.WriteLine("=============== End of SVM training ===============");
            Console.WriteLine();

            return model;
        }

        private static void EvaluateNB(MLContext mlContext, ITransformer model, IDataView trainingData)
        {
            IDataView predictions = model.Transform(trainingData);

            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("NB-Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy:F2}");
            Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy:F2}");
            Console.WriteLine($"Log Loss: {metrics.LogLoss:F2}");
            Console.WriteLine($"Log Loss Reduction: {metrics.LogLossReduction:F2}\n");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine("=============== End of NB-model evaluation ===============");
        }
        private static void EvaluateSVM(MLContext mlContext, ITransformer model, IDataView trainingData)
        {
            IDataView predictions = model.Transform(trainingData);

            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(data: predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("SVM-Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of SVM-model evaluation ===============");
        }

        private static void ValidateNB(MLContext mlContext, IDataView data)
        {
            var estimator = mlContext.Transforms
                          .Conversion.MapValueToKey("Label")
                          .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "Features"))
                          .Append(mlContext.MulticlassClassification.Trainers.NaiveBayes(labelColumnName: "Label", featureColumnName: "Features"));

            Console.WriteLine();
            Console.WriteLine("NB-Model validation metrics");
            Console.WriteLine("--------------------------------");

            var results = mlContext.MulticlassClassification.CrossValidate(data, estimator, numberOfFolds: 10);

            var micro = results.Average(fold => fold.Metrics.MicroAccuracy);
            var macro = results.Average(fold => fold.Metrics.MacroAccuracy);


            Console.WriteLine($"AVG Micro Accuracy: {micro}");
            Console.WriteLine($"AVG Macro Accuracy: {macro}");
            Console.WriteLine("=============== End of NB-model validation ===============");
            Console.WriteLine();
        }

        private static void ValidateSVM(MLContext mlContext, IDataView data)
        {
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "Features")
                         .Append(mlContext.BinaryClassification.Trainers.LinearSvm("Label", featureColumnName: "Features"));

            Console.WriteLine();
            Console.WriteLine("SVM-Model validation metrics");
            Console.WriteLine("--------------------------------");
            var crossvalidator = mlContext.BinaryClassification.CrossValidateNonCalibrated(data, estimator, numberOfFolds: 10);

            var acc = crossvalidator.Average(f => f.Metrics.Accuracy);
            var auc = crossvalidator.Average(f => f.Metrics.AreaUnderRocCurve);


            Console.WriteLine($"AVG Accuracy: {acc:P2}");
            Console.WriteLine($"AVG AUC: {auc:P2}");
            Console.WriteLine("=============== End of SVM-model validation ===============");
            Console.WriteLine();
        }
        private static void TestNB(MLContext mLContext, List<TextML> Texts)
        {

        }

        private static void TestSdca(MLContext mlContext, ITransformer model, List<TextML> Texts)
        {
            PredictionEngine<TextDataNB, TextPredictionNB> predictionFunction = mlContext.Model.CreatePredictionEngine<TextDataNB, TextPredictionNB>(model);


            int exampleB = 0; //pobieram recenzję, która ma najgorszą ocenę wśród innych
            int exampleG = 1200; //tu pobieram recenzję z końca listy, czyli która ma jedną z najlepszych ocen
            int exampleN = 650; // biorę średnią recenzję ze średnią oceną
            var sampleStatementB = new TextDataNB() { TextSubj = Texts.Last().TextLinesArr[exampleB] };
            var sampleStatementG = new TextDataNB() { TextSubj = Texts.Last().TextLinesArr[exampleG] };
            var sampleStatementN = new TextDataNB() { TextSubj = Texts.Last().TextLinesArr[exampleN] };

            var resultPredictionB = predictionFunction.Predict(sampleStatementB);
            var resultPredictionG = predictionFunction.Predict(sampleStatementG);
            var resultPredictionN = predictionFunction.Predict(sampleStatementN);
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");

            Console.WriteLine();
            Console.WriteLine($"Rating: {Texts.Last().ReviewMLs.ElementAt(exampleB).Rating} | " +
                              $"Prediction: {(Convert.ToBoolean(resultPredictionB.Prediction) ? "Negative" : "Positive")} | " +
                              $"Probability: {resultPredictionB.Probability} ");
            Console.WriteLine($"Rating: {Texts.Last().ReviewMLs.ElementAt(exampleG).Rating} | " +
                              $"Prediction: {(Convert.ToBoolean(resultPredictionG.Prediction) ? "Negative" : "Positive")} | " +
                              $"Probability: {resultPredictionG.Probability} ");
            Console.WriteLine($"Rating: {Texts.Last().ReviewMLs.ElementAt(exampleN).Rating} | " +
                              $"Prediction: {(Convert.ToBoolean(resultPredictionN.Prediction) ? "Negative" : "Positive")} | " +
                              $"Probability: {resultPredictionN.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();

            #region DaneTestoweDoEksperymetowWPrzyszlosci
            /*
                        IEnumerable<TextData> reviews = new[]
                          {
                                new TextData
                                {
                                    TextSubj = "This was a very bad movie"
                                },
                                new TextData
                                {
                                    TextSubj = "I love that movie"
                                },
                          };
                        IDataView batchComments = mlContext.Data.LoadFromEnumerable(reviews);

                        IDataView predictions = model.Transform(batchComments);

                        IEnumerable<TextPrediction> predictedResults = mlContext.Data.CreateEnumerable<TextPrediction>(predictions, reuseRowObject: false);

                        Console.WriteLine();
                        Console.WriteLine("=============== Prediction Test of loaded model with a multiple samples ===============");

                        Console.WriteLine();
                        foreach (var prediction in predictedResults)
                            Console.WriteLine($"Sentiment: {prediction.TextSubj} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");

                        Console.WriteLine("=============== End of Predictions ===============");
                        Console.WriteLine();
            */
            #endregion
        }

        private static TextML GetFile(string name)
        {
            string folderPath = Path.Combine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName, "scaledata", name);
            string[] idFile = File.ReadAllLines(Path.Combine(folderPath, $"id.{name}"));
            string[] label3classFile = File.ReadAllLines(Path.Combine(folderPath, $"label.3class.{name}"));
            string[] label4classFile = File.ReadAllLines(Path.Combine(folderPath, $"label.4class.{name}"));
            string[] ratingFile = File.ReadAllLines(Path.Combine(folderPath, $"rating.{name}"));
            string[] subjFile = File.ReadAllLines(Path.Combine(folderPath, $"subj.{name}"));

            TextML textML = new TextML(name, idFile, label3classFile, label4classFile, ratingFile, subjFile);

            return textML;
        }
    }
}