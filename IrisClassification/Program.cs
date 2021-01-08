using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace IrisClassification {
  class Program {
    // STEP 1: Define your data structures

    // IrisData is used to provide training data, and as
    // input for prediction operations
    // - First 4 properties are input/features used to predict the label
    // - Label is what you are predicting, and is only set when training
    public class IrisData {
      [Column("0")]
      public float SepalLength;

      [Column("1")]
      public float SepalWidth;

      [Column("2")]
      public float PetalLength;

      [Column("3")]
      public float PetalWidth;

      [Column("4")]
      public string Label;
    }

    // IrisPrediction is the result returned from prediction operations
    public class IrisPrediction {
      [ColumnName("PredictedLabel")]
      public string PredictedLabels;
    }

    static void Main(string[] args) {
      // STEP 2: Create a pipeline and load your data
      var pipeline = new LearningPipeline();

      string dataPath;
      Console.Write("Enter data path. If no path is provided, the program will look for \"iris-data.txt\" in the current working directory. Enter period if you do not wish to provide path: ");
      Console.ForegroundColor = ConsoleColor.Green;
      dataPath = Console.ReadLine();
      Console.ForegroundColor = ConsoleColor.White;
      Console.WriteLine("");
      if (String.Equals(dataPath, ".")) {
        dataPath = "iris-data.txt";
      }

      // Add exception
      AppDomain.CurrentDomain.UnhandledException += new UnhandledExceptionEventHandler(CurrentDomain_UnhandledException);

      pipeline.Add(new TextLoader < IrisData > (dataPath, separator: ","));

      // STEP 3: Transform your data
      // Assign numeric values to text in the "Label" column, because only
      // numbers can be processed during model training
      pipeline.Add(new Dictionarizer("Label"));

      // Puts all features into a vector
      pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

      // STEP 4: Add learner
      // Add a learning algorithm to the pipeline.
      // This is a classification scenario (What type of iris is this?)
      pipeline.Add(new StochasticDualCoordinateAscentClassifier());

      // Convert the Label back into original text (after converting to number in step 3)
      pipeline.Add(new PredictedLabelColumnOriginalValueConverter() {
        PredictedLabelColumn = "PredictedLabel"
      });

      // STEP 5: Train your model based on data set
      Console.ForegroundColor = ConsoleColor.Blue;
      Console.WriteLine("Training the model...");
      var model = pipeline.Train < IrisData,
        IrisPrediction > ();

      // STEP 6: Use your model to make a prediction
      // You can change these numbers to test different predictions
      Console.ForegroundColor = ConsoleColor.Green;
      Console.WriteLine("Training complete.\n");
      Console.ForegroundColor = ConsoleColor.White;

      float sLength, sWidth, pLength, pWidth;
      Console.Write("Enter Sepal Length: ");
      sLength = float.Parse(Console.ReadLine());
      Console.Write("Enter Sepal Width: ");
      sWidth = float.Parse(Console.ReadLine());
      Console.Write("Enter Petal Length: ");
      pLength = float.Parse(Console.ReadLine());
      Console.Write("Enter Petal Width: ");
      pWidth = float.Parse(Console.ReadLine());

      var prediction = model.Predict(new IrisData() {
        SepalLength = sLength,
          SepalWidth = sWidth,
          PetalLength = pLength,
          PetalWidth = pWidth,
      });

      Console.Write("Predicted flower type is: ");
      Console.ForegroundColor = ConsoleColor.Magenta;
      Console.WriteLine($"{prediction.PredictedLabels}");
    }

    static void CurrentDomain_UnhandledException(object sender, UnhandledExceptionEventArgs e) {
      Console.ForegroundColor = ConsoleColor.Red;
      Console.WriteLine((e.ExceptionObject as Exception).Message);
    }

  }
}
