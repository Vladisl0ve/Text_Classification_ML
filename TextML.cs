using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using ConsoleTables;

namespace Text_Classification_ML
{
    class ReviewML
    {
        public int Id { get; set; } 
        public string Subj { get; set; } // Tekst recenzji
        public int Label3Class { get; set; } // {0, 1, 2}
        public int Label4Class { get; set; } // {0, 1, 2, 3}
        public float Rating { get; set; } //Ocena, na podstawie której zostały stworzone Label3 oraz Label4

        public bool IsToxic
        {
            get => Label4Class == 0 || Label4Class == 1; // {0, 1} - opinia jest negatywną
                                                         // {2, 3} - opinia jest pozytywną
        }

        public ReviewML(string id, string label3class, string label4class, string rating, string subj)
        {
            Id = int.Parse(id);
            Label3Class = int.Parse(label3class);
            Label4Class = int.Parse(label4class);
            Rating = float.Parse(rating, CultureInfo.InvariantCulture);
            Subj = subj;
        }
    }

    class TextML
    {
        public string Name;

        public List<ReviewML> ReviewMLs = new List<ReviewML>();

        private string[] _id;
        private string[] _label3class;
        private string[] _label4class;
        private string[] _rating;
        private string[] _subj;

        public TextML(string name, string[] id, string[] label3class, string[] label4class, string[] rating, string[] subj)
        {
            Name = name;

            _id = id;
            _label3class = label3class;
            _label4class = label4class;
            _rating = rating;
            _subj = subj;

            InitReviewML();
            InitInfo();
        }

        private void InitReviewML()
        {
            for (int i = 0; i < _id.Length; i++)
                ReviewMLs.Add(new ReviewML(_id[i], _label3class[i], _label4class[i], _rating[i], _subj[i]));
        }

        private void InitInfo()
        {
            if (_subj.Count() == 0)
                return;

            foreach (var word in Words)
            {
                if (!InfoWordCounter.ContainsKey(word))
                    InfoWordCounter.Add(word, 1);
                else
                    InfoWordCounter[word]++;
            }

            InfoWordCounter = InfoWordCounter.OrderByDescending(i => i.Value).ThenBy(i => i.Key).ToDictionary(x => x.Key, x => x.Value);

        }

        public string[] TextLinesArr { get => _subj; }
        public string[] Words
        {
            get
            {
                List<string> words = new List<string>();
                foreach (var line in _subj)
                {
                    words.AddRange(GetWords(line));
                }

                return words.ToArray();
            }
        }
        Dictionary<string, int> InfoWordCounter = new Dictionary<string, int>();
        public TextML(string name, string[] subj)
        {
            Name = name;
            _subj = subj;
        }



        public void ShowTop10Rarest()
        {
            ConsoleTable table = new ConsoleTable("Lp.", "Word", "Amount");
            int last = InfoWordCounter.Count() - 1;

            table.AddRow(5, InfoWordCounter.ElementAt(last - 4).Key, InfoWordCounter.ElementAt(last - 4).Value);
            table.AddRow(4, InfoWordCounter.ElementAt(last - 3).Key, InfoWordCounter.ElementAt(last - 3).Value);
            table.AddRow(3, InfoWordCounter.ElementAt(last - 2).Key, InfoWordCounter.ElementAt(last - 2).Value);
            table.AddRow(2, InfoWordCounter.ElementAt(last - 1).Key, InfoWordCounter.ElementAt(last - 1).Value);
            table.AddRow(1, InfoWordCounter.ElementAt(last).Key, InfoWordCounter.ElementAt(last).Value);

            Console.WriteLine("---------- TOP 10 RAREST WORDS ----------\n");
            Console.WriteLine(table.ToStringAlternative());
        }

        public void ShowTop10Popular()
        {
            ConsoleTable table = new ConsoleTable("Lp.", "Word", "Amount");

            table.AddRow(1, InfoWordCounter.ElementAt(0).Key, InfoWordCounter.ElementAt(0).Value);
            table.AddRow(2, InfoWordCounter.ElementAt(1).Key, InfoWordCounter.ElementAt(1).Value);
            table.AddRow(3, InfoWordCounter.ElementAt(2).Key, InfoWordCounter.ElementAt(2).Value);
            table.AddRow(4, InfoWordCounter.ElementAt(3).Key, InfoWordCounter.ElementAt(3).Value);
            table.AddRow(5, InfoWordCounter.ElementAt(4).Key, InfoWordCounter.ElementAt(4).Value);

            Console.WriteLine("---------- TOP 10 POPULAR WORDS ----------\n");
            Console.WriteLine(table.ToStringAlternative());
        }
        public void ShowSize() => Console.WriteLine($"AMOUNT OF LINES: {_subj.Length}\n" +
                                                    $"AMOUNT OF WORDS: {Words.Length}\n" +
                                                     $"AMOUNT OF UNIQUE WORDS: {InfoWordCounter.Count()}");

        public void ShowMoreStats()
        {
            string[] uniqWords = InfoWordCounter.Keys.ToArray();
            string longest = uniqWords.OrderByDescending(w => w.Length).First();

            Console.WriteLine($"The longest word is {longest}");
        }

        public void ShowStats()
        {
            Console.WriteLine($"################ {Name} ################");
            Console.WriteLine($"");
            ShowTop10Popular();
            Console.WriteLine($"");
            ShowSize();
            Console.WriteLine($"");
            ShowTop10Rarest();
            Console.WriteLine($"");
            ShowMoreStats();
            Console.WriteLine($"");
        }

        public void WriteStatsToFile()
        {

            ConsoleTable table = new ConsoleTable("Lp.", "Word", "Amount");

            for (int i = 0; i < InfoWordCounter.Count(); i++)
                table.AddRow(i + 1, InfoWordCounter.ElementAt(i).Key, InfoWordCounter.ElementAt(i).Value);

            string toWrite = $"{Name}\n" +
                             $"{table.ToStringAlternative()}";
            string pathToSave = Path.Combine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName,
                                "Saved_results",
                                 $"{Name}_stats.txt");
            File.WriteAllText(pathToSave, toWrite);

        }

        static string[] GetWords(string input)
        {
            MatchCollection matches = Regex.Matches(input, @"\b[\w']*\b");

            var words = from m in matches.Cast<Match>()
                        where !string.IsNullOrEmpty(m.Value)
                        where !m.Value.Contains('_')
                        select TrimSuffix(m.Value);

            return words.ToArray();
        }

        static string TrimSuffix(string word)
        {
            int apostropheLocation = word.IndexOf('\'');
            if (apostropheLocation != -1)
            {
                word = word.Substring(0, apostropheLocation);
            }

            return word;
        }
    }
}
