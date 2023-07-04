using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using Accord.IO;

namespace Program
{
    class Program
    {
        static void Main(string[] args)
        {
            string dataDir = "./input/";
            string edfDir = Path.Combine(dataDir, "edf_data");

            string trainRecordsPath = Path.Combine(dataDir, "train_records.csv");
            string testRecordsPath = Path.Combine(dataDir, "test_records.csv");

            // パスを設定
            var trainRecordData = File.ReadAllLines(trainRecordsPath)
                .Skip(1) // Skip header
                .Select(line => line.Split(','))
                .Select(parts => new
                {
                    Hypnogram = Path.Combine(edfDir, parts[0]),
                    Psg = Path.Combine(edfDir, parts[1])
                })
                .ToList();

            var testRecordData = File.ReadAllLines(testRecordsPath)
                .Skip(1) // Skip header
                .Select(line => line.Split(','))
                .Select(parts => new
                {
                    Psg = Path.Combine(edfDir, parts[0])
                })
                .ToList();

            var row = trainRecordData.First();

            // edfファイルの読み込み
            var psgEdf = new RawEDFFile(row.Psg, false);

            // 読み込んだデータは、RawEDFFileクラスのインスタンスになります
            Type type = psgEdf.GetType();
            Console.WriteLine(type);

            // infoでメタ情報を表示できます
            var info = psgEdf.Info;
            Console.WriteLine(info);
        }
    }
}
