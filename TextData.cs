using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace Text_Classification_ML
{
    public class TextData
    {
        [ColumnName("Label")]
        public bool TextIsToxic { get; set; }
        public string TextSubj { get; set; }

    }

    public class TextPrediction : TextData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }
}
