using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Globalization;
using System.Collections.Generic;
using back_propagation;


Main();
static void Main()
{
    Console.WriteLine("Enter command ('train' or 'test'):");
    var cmd = Console.ReadLine()?.Trim().ToLowerInvariant();
    if (cmd == "train")
    {
        Train();
    }
    else if (cmd == "test")
    {
        Test();
    }
    else
    {
        Console.WriteLine("Unknown command. Exiting.");
    }
}

static decimal ReluPrimeFromActivation(decimal a)
{
    // a = ReLU(z)이면 a>0 <=> z>0 이므로 이 마스크로 도함수 처리 가능(0은 관례적으로 0 처리)
    return a > 0 ? 1m : 0m;
}

static decimal Relu(decimal x)
{
    return x > 0m ? x : 0m;
}

static decimal NormalizedPixelValue(Color pixelColor)
{
    // MNIST는 흰 배경/검은 글자이므로 반전해 글자(전경)가 큰 값이 되도록 맞춘다.
    decimal gray = Convert.ToDecimal(((0.299 * pixelColor.R) + (0.587 * pixelColor.G) + (0.114 * pixelColor.B)) / 255.0);
    return 1m - gray;
}

static decimal XavierUniformWeight(Random rand, int fanIn)
{
    // 너무 큰 초기값으로 출력이 폭주하지 않도록 fan-in 기반으로 초기화 범위를 제한
    double limit = Math.Sqrt(6.0 / fanIn);
    double sampled = ((rand.NextDouble() * 2.0) - 1.0) * limit;
    return Convert.ToDecimal(sampled);
}

static int ReadPositiveEnvInt(string key)
{
    string? raw = Environment.GetEnvironmentVariable(key);
    if (int.TryParse(raw, out int parsed) && parsed > 0)
    {
        return parsed;
    }

    return 0;
}

static void Train()
{
    DirectoryInfo di = new DirectoryInfo(System.Environment.CurrentDirectory + @"/mnist_png/training");
    List<ImageData> imageDatas = new List<ImageData>();
    foreach (var subdir in di.GetDirectories())
    {
        Console.WriteLine(subdir.FullName);
        foreach (var file in subdir.GetFiles())
        {
            imageDatas.Add(new ImageData(file.FullName, int.Parse(subdir.Name), int.Parse(file.Name.Split('.')[0])));
        }
    }

    imageDatas.Sort(((a, b) => { if (a.FileName == b.FileName) return 0; else return (a.FileName < b.FileName) ? 1 : -1; }));
    int maxTrainSamples = ReadPositiveEnvInt("MAX_TRAIN_SAMPLES");
    if (maxTrainSamples > 0 && maxTrainSamples < imageDatas.Count)
    {
        imageDatas = imageDatas.Take(maxTrainSamples).ToList();
        Console.WriteLine($"Training sample limit enabled: {imageDatas.Count}");
    }

    Node[] input = new Node[28 * 28];
    Node[] hidden1 = new Node[128];
    Node[] hidden2 = new Node[128];
    Node[] output = new Node[10];
    Random rand = new Random(DateTime.Now.Millisecond);
    decimal updateRate = 0.001m;
    for (int i = 0; i < 28 * 28; i++)
    {
        decimal[] w = new decimal[128];
        for (int j = 0; j < 128; j++)
        {
            w[j] = XavierUniformWeight(rand, 28 * 28);
        }
        input[i] = new Node(w);
    }

    for (int i = 0; i < 128; i++)
    {
        decimal[] w = new decimal[128];
        for (int j = 0; j < 128; j++)
        {
            w[j] = XavierUniformWeight(rand, 128);
        }
        hidden1[i] = new Node(w);
    }

    for (int i = 0; i < 128; i++)
    {
        decimal[] w = new decimal[10];
        for (int j = 0; j < 10; j++)
        {
            w[j] = XavierUniformWeight(rand, 128);
        }
        hidden2[i] = new Node(w);
    }

    for (int i = 0; i < 10; i++)
    {
        output[i] = new Node();
    }

    for (int epoch = 0; epoch < 10; epoch++)
    {
        Console.WriteLine($"Epoch {epoch + 1}/10");
        foreach (var imageData in imageDatas)
        {
            Bitmap bitmap = new Bitmap(imageData.ImagePath);

            for (int i = 0; i < bitmap.Width; i++)
            {
                for (int j = 0; j < bitmap.Height; j++)
                {
                    Color pixelColor = bitmap.GetPixel(i, j);
                    input[i * 28 + j].x = NormalizedPixelValue(pixelColor);
                }
            }
            for (int h1 = 0; h1 < hidden1.Length; h1++)
            {
                hidden1[h1].Calculate(input, h1);
            }
            for (int h2 = 0; h2 < hidden2.Length; h2++)
            {
                hidden2[h2].CalculateReLU(hidden1, h2);
            }
            for (int o = 0; o < output.Length; o++)
            {
                output[o].CalculateReLU(hidden2, o);
            }
            decimal[] T = new decimal[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            T[imageData.Label] = 1;
            decimal[] deltaOut = new decimal[10];
            for (int o = 0; o < 10; o++)
            {
                deltaOut[o] = (2m / 10m) * (output[o].x - T[o]);
            }

            decimal[] delta2 = new decimal[hidden2.Length];
            for (int h2 = 0; h2 < hidden2.Length; h2++)
            {
                decimal sum = 0;
                for (int o = 0; o < 10; o++)
                {
                    sum += hidden2[h2].weights[o] * deltaOut[o];
                }
                delta2[h2] = sum * ReluPrimeFromActivation(hidden2[h2].x);
            }

            decimal[] delta1 = new decimal[hidden1.Length];
            for (int h1 = 0; h1 < hidden1.Length; h1++)
            {
                decimal sum = 0;
                for (int h2 = 0; h2 < hidden2.Length; h2++)
                {
                    sum += hidden1[h1].weights[h2] * delta2[h2];
                }
                delta1[h1] = sum * ReluPrimeFromActivation(hidden1[h1].x);
            }

            for (int h2 = 0; h2 < hidden2.Length; h2++)
            {
                for (int o = 0; o < 10; o++)
                {
                    hidden2[h2].weights[o] -= updateRate * Relu(hidden2[h2].x) * deltaOut[o];
                }
            }

            for (int h1 = 0; h1 < hidden1.Length; h1++)
            {
                for (int h2 = 0; h2 < hidden2.Length; h2++)
                {
                    hidden1[h1].weights[h2] -= updateRate * Relu(hidden1[h1].x) * delta2[h2];
                }
            }

            for (int inp = 0; inp < input.Length; inp++)
            {
                for (int h1 = 0; h1 < hidden1.Length; h1++)
                {
                    input[inp].weights[h1] -= updateRate * input[inp].x * delta1[h1];
                }
            }
            Console.WriteLine($"epoch#{epoch + 1} {imageDatas.IndexOf(imageData) + 1}/{imageDatas.Count}");
        }
    }
    {
        string modelPath = Path.Combine(Environment.CurrentDirectory, "model_weights.txt");
        using (var sw = new StreamWriter(modelPath, false))
        {
            sw.WriteLine("# INPUT");
            foreach (var n in input)
            {
                sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
            }

            sw.WriteLine("# HIDDEN1");
            foreach (var n in hidden1)
            {
                sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
            }

            sw.WriteLine("# HIDDEN2");
            foreach (var n in hidden2)
            {
                sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
            }

            sw.WriteLine("# OUTPUT");
            foreach (var n in output)
            {
                sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
            }
        }
    }
}

static void Test()
{
    // Prepare nodes structures
    Node[] input = new Node[28 * 28];
    Node[] hidden1 = new Node[128];
    Node[] hidden2 = new Node[128];
    Node[] output = new Node[10];

    // Initialize with default arrays to be replaced by loaded weights
    for (int i = 0; i < input.Length; i++) input[i] = new Node(new decimal[128]);
    for (int i = 0; i < hidden1.Length; i++) hidden1[i] = new Node(new decimal[128]);
    for (int i = 0; i < hidden2.Length; i++) hidden2[i] = new Node(new decimal[10]);
    for (int i = 0; i < output.Length; i++) output[i] = new Node();

    // Load model weights
    string modelPath = Path.Combine(Environment.CurrentDirectory, "model_weights.txt");
    if (!File.Exists(modelPath))
    {
        Console.WriteLine("model_weights.txt not found. Please run 'train' first.");
        return;
    }

    var lines = File.ReadAllLines(modelPath).Select(l => l.Trim()).Where(l => l.Length > 0).ToList();
    string section = "";
    int inpIndex = 0, h1Index = 0, h2Index = 0, outIndex = 0;
    foreach (var line in lines)
    {
        if (line.StartsWith("#"))
        {
            if (line.Contains("INPUT")) section = "INPUT";
            else if (line.Contains("HIDDEN1")) section = "HIDDEN1";
            else if (line.Contains("HIDDEN2")) section = "HIDDEN2";
            else if (line.Contains("OUTPUT")) section = "OUTPUT";
            continue;
        }

        var parts = line.Split(',');
        try
        {
            if (section == "INPUT" && inpIndex < input.Length)
            {
                input[inpIndex].weights = parts.Select(p => decimal.Parse(p, CultureInfo.InvariantCulture)).ToArray();
                inpIndex++;
            }
            else if (section == "HIDDEN1" && h1Index < hidden1.Length)
            {
                hidden1[h1Index].weights = parts.Select(p => decimal.Parse(p, CultureInfo.InvariantCulture)).ToArray();
                h1Index++;
            }
            else if (section == "HIDDEN2" && h2Index < hidden2.Length)
            {
                hidden2[h2Index].weights = parts.Select(p => decimal.Parse(p, CultureInfo.InvariantCulture)).ToArray();
                h2Index++;
            }
            else if (section == "OUTPUT" && outIndex < output.Length)
            {
                output[outIndex].weights = parts.Select(p => decimal.Parse(p, CultureInfo.InvariantCulture)).ToArray();
                outIndex++;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to parse weights line: {line}. Exception: {ex.Message}");
        }
    }

    // Load testing images
    DirectoryInfo di = new DirectoryInfo(System.Environment.CurrentDirectory + @"/mnist_png/testing");
    List<ImageData> imageDatas = new List<ImageData>();
    foreach (var subdir in di.GetDirectories())
    {
        foreach (var file in subdir.GetFiles())
        {
            imageDatas.Add(new ImageData(file.FullName, int.Parse(subdir.Name), int.Parse(file.Name.Split('.')[0].Split('a')[1])));
        }
    }

    imageDatas.Sort(((a, b) => { if (a.FileName == b.FileName) return 0; else return (a.FileName < b.FileName) ? 1 : -1; }));
    int maxTestSamples = ReadPositiveEnvInt("MAX_TEST_SAMPLES");
    if (maxTestSamples > 0 && maxTestSamples < imageDatas.Count)
    {
        imageDatas = imageDatas.Take(maxTestSamples).ToList();
        Console.WriteLine($"Test sample limit enabled: {imageDatas.Count}");
    }

    int correct = 0;
    int total = imageDatas.Count;
    for (int idxImg = 0; idxImg < imageDatas.Count; idxImg++)
    {
        var imageData = imageDatas[idxImg];
        Bitmap bitmap = new Bitmap(imageData.ImagePath);
        for (int i = 0; i < bitmap.Width; i++)
        {
            for (int j = 0; j < bitmap.Height; j++)
            {
                Color pixelColor = bitmap.GetPixel(i, j);
                input[i * 28 + j].x = NormalizedPixelValue(pixelColor);
            }
        }

        for (int h1 = 0; h1 < hidden1.Length; h1++)
        {
            hidden1[h1].Calculate(input, h1);
        }

        for (int h2 = 0; h2 < hidden2.Length; h2++)
        {
            hidden2[h2].CalculateReLU(hidden1, h2);
        }

        for (int o = 0; o < output.Length; o++)
        {
            output[o].CalculateReLU(hidden2, o);
        }

        // prediction
        int pred = 0;
        decimal max = output[0].x;
        for (int k = 1; k < output.Length; k++)
        {
            if (output[k].x > max) { max = output[k].x; pred = k; }
        }
        if (pred == imageData.Label) correct++;

        Console.WriteLine($"{idxImg + 1}/{total} - label: {imageData.Label}, pred: {pred}");
    }

    Console.WriteLine($"Test complete. Accuracy: {(double)correct / total:P2} ({correct}/{total})");
}
