using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace Text_Classification_ML
{
    public class TextDataNB
    {
        public bool TextIsToxic { get; set; }
        public int Label3Class { get; set; }
        public int Label4Class { get; set; }
        public float Rating { get; set; }
        [ColumnName("Label")]
        public float TextIsToxicInt { get => TextIsToxic ? 1 : 0; }
        [ColumnName("Features")]
        public string TextSubj { get; set; }

    }

    public class TextPredictionNB : TextDataNB
    {
        [ColumnName("PredictedLabel")]
        public uint Prediction { get; set; }
        [ColumnName("Probability")]
        public float Probability { get; set; }
        public float Score { get; set; }
    }
}
