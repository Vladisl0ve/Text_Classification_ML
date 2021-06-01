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
                                       // "Scott+Renshaw",
                                        //"Steve+Rhodes",
                                    };

            List<TextML> Texts = new List<TextML>();
            foreach (var name in FilesToClassify)
                Texts.Add(GetFile(name));

            List<String> Files1 = new List<string>() { FilesToClassify.First() };
            List<String> Files2 = new List<string>() { FilesToClassify.Last() };
            var mlContext = new MLContext();

            var trainData = LoadData(mlContext, Texts, FilesToClassify);
            var model = BuildAndTrainModel(mlContext, trainData);

            Evaluate(mlContext, model, trainData);

            Test(mlContext, model, Texts);


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

        private static IDataView LoadData(MLContext mlContext, List<TextML> listTexts, List<string> files)
        {
            List<TextData> textData = new List<TextData>();

            foreach (var file in files)
            {
                foreach (var review in listTexts.Where(t => t.Name == file).ToList().First().ReviewMLs)
                {
                    textData.Add(new TextData() { TextIsToxic = review.IsToxic, TextSubj = review.Subj });
                }
            }

            IDataView trainingData = mlContext.Data.LoadFromEnumerable(textData);
            return trainingData;
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingData)
        {
            var estimator = mlContext.Transforms.Text
                          .FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(TextData.TextSubj))
                          .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(trainingData);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model, IDataView trainingData)
        {
            IDataView predictions = model.Transform(trainingData);

            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        private static void Test(MLContext mlContext, ITransformer model, List<TextML> Texts)
        {
            PredictionEngine<TextData, TextPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<TextData, TextPrediction>(model);


            int exampleB = 0; //pobieram recenzję, która ma najgorszą ocenę wśród innych
            int exampleG = 1200; //tu pobieram recenzję z końca listy, czyli która ma jedną z najlepszych ocen
            int exampleN = 650; // biorę średnią recenzję ze średnią oceną
            var sampleStatementB = new TextData() { TextSubj = Texts.Last().TextLinesArr[exampleB] };
            var sampleStatementG = new TextData() { TextSubj = Texts.Last().TextLinesArr[exampleG] };
            var sampleStatementN = new TextData() { TextSubj = Texts.Last().TextLinesArr[exampleN] };

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