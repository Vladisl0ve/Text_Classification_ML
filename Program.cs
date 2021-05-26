using System;
using System.Collections.Generic;
using System.IO;

namespace Text_Classification_ML
{
    class Program
    {
        static void Main(string[] args)
        {
            List<string> FilesToClassify = new List<string>()
            {
                "Dennis+Schwartz",
                "James+Berardinelli",
                "Scott+Renshaw",
                "Steve+Rhodes",
            };

            List<TextML> Texts = new List<TextML>();
            foreach (var name in FilesToClassify)
                Texts.Add(GetFile(name));

            Texts.ForEach(t => t.ShowStats());

            //Console.WriteLine(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "scaledata", "2", ""));
            Console.ReadKey();
        }

        static TextML GetFile(string name)
        {
            string folderPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "scaledata", name);
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
