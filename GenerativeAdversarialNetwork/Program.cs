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
    Console.WriteLine("Enter command ('train', 'dtest', or 'gtest'):");
    var cmd = Console.ReadLine()?.Trim().ToLowerInvariant();
    if (cmd == "train")
    {
        Train();
    }
    else if (cmd == "dtest" || cmd == "test")
    {
        DTest();
    }
    else if (cmd == "gtest")
    {
        GTest();
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

const decimal UpdateActivationClip = 10m;
const decimal UpdateDeltaClip = 10m;
const decimal UpdateGradientClip = 25m;
const decimal WeightClip = 5m;

static decimal Clamp(decimal value, decimal min, decimal max)
{
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

static decimal Sigmoid(decimal x)
{
    double z = Math.Clamp(Convert.ToDouble(x), -40.0, 40.0);
    double y = 1.0 / (1.0 + Math.Exp(-z));
    return Convert.ToDecimal(y);
}

static decimal ApplyWeightUpdate(decimal weight, decimal activation, decimal delta, decimal learningRate)
{
    decimal clippedActivation = Clamp(activation, -UpdateActivationClip, UpdateActivationClip);
    decimal clippedDelta = Clamp(delta, -UpdateDeltaClip, UpdateDeltaClip);
    decimal gradient = Clamp(clippedActivation * clippedDelta, -UpdateGradientClip, UpdateGradientClip);
    decimal updated = weight - (learningRate * gradient);
    return Clamp(updated, -WeightClip, WeightClip);
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

    Node[] D_input = new Node[28 * 28];
    Node[] D_hidden1 = new Node[128];
    Node[] D_hidden2 = new Node[128];
    Node[] D_output = new Node[11];
    Node[] G_input = new Node[10];
    Node[] G_hidden1 = new Node[128];
    Node[] G_hidden2 = new Node[128];
    Node[] G_output = new Node[28 * 28];
    Random rand = new Random(DateTime.Now.Millisecond);
    decimal updateRate = 0.001m;
    for (int i = 0; i < 28 * 28; i++)
    {
        decimal[] w = new decimal[128];
        for (int j = 0; j < 128; j++)
        {
            w[j] = XavierUniformWeight(rand, 28 * 28);
        }
        D_input[i] = new Node(w);
    }

    for (int i = 0; i < 128; i++)
    {
        decimal[] w = new decimal[128];
        for (int j = 0; j < 128; j++)
        {
            w[j] = XavierUniformWeight(rand, 128);
        }
        D_hidden1[i] = new Node(w);
    }

    for (int i = 0; i < 128; i++)
    {
        decimal[] w = new decimal[11];
        for (int j = 0; j < 11; j++)
        {
            w[j] = XavierUniformWeight(rand, 128);
        }
        D_hidden2[i] = new Node(w);
    }

    for (int i = 0; i < 11; i++)
    {
        D_output[i] = new Node();
    }

    for (int i = 0; i < 10; i++)
    {
        decimal[] w = new decimal[128];
        for (int j = 0; j < 128; j++)
        {
            w[j] = XavierUniformWeight(rand, 10);
        }
        G_input[i] = new Node(w);
    }

    for (int i = 0; i < 128; i++)
    {
        decimal[] w = new decimal[128];
        for (int j = 0; j < 128; j++)
        {
            w[j] = XavierUniformWeight(rand, 128);
        }
        G_hidden1[i] = new Node(w);
    }

    for (int i = 0; i < 128; i++)
    {
        decimal[] w = new decimal[28 * 28];
        for (int j = 0; j < 28 * 28; j++)
        {
            w[j] = XavierUniformWeight(rand, 128);
        }
        G_hidden2[i] = new Node(w);
    }

    for (int i = 0; i < 28 * 28; i++)
    {
        G_output[i] = new Node();
    }
    int epochs = 10;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        Console.WriteLine($"Epoch {epoch + 1}/{epochs}");
        foreach (var imageData in imageDatas)
        {
            Bitmap bitmap = new Bitmap(imageData.ImagePath);

            for (int i = 0; i < bitmap.Width; i++)
            {
                for (int j = 0; j < bitmap.Height; j++)
                {
                    Color pixelColor = bitmap.GetPixel(i, j);
                    D_input[i * 28 + j].x = NormalizedPixelValue(pixelColor);
                }
            }
            int generatedDigit = rand.Next(10);
            for (int i = 0; i < G_input.Length; i++)
            {
                G_input[i].x = i == generatedDigit ? 1m : 0m;
            }
            for (int h1 = 0; h1 < G_hidden1.Length; h1++)
            {
                G_hidden1[h1].Calculate(G_input, h1);
            }
            for (int h2 = 0; h2 < G_hidden2.Length; h2++)
            {
                G_hidden2[h2].CalculateReLU(G_hidden1, h2);
            }
            for (int o = 0; o < G_output.Length; o++)
            {
                G_output[o].Calculate(G_hidden2, o);
                G_output[o].x = Sigmoid(G_output[o].x); // sigmoid으로 확률화
            }
            for (int h1 = 0; h1 < D_hidden1.Length; h1++)
            {
                D_hidden1[h1].Calculate(D_input, h1);
            }
            for (int h2 = 0; h2 < D_hidden2.Length; h2++)
            {
                D_hidden2[h2].CalculateReLU(D_hidden1, h2);
            }
            for (int o = 0; o < D_output.Length; o++)
            {
                D_output[o].Calculate(D_hidden2, o);
                D_output[o].x = Sigmoid(D_output[o].x); // sigmoid으로 확률화
            }
            // D(real): 정답 숫자(0~9) one-hot 타깃으로 학습
            decimal[] deltaOutReal = new decimal[D_output.Length];
            for (int o = 0; o < D_output.Length; o++)
            {
                decimal target = (o == imageData.Label) ? 1m : 0m;
                deltaOutReal[o] = D_output[o].x - target;
            }

            decimal[] delta2Real = new decimal[D_hidden2.Length];
            for (int h2 = 0; h2 < D_hidden2.Length; h2++)
            {
                decimal sum = 0m;
                for (int o = 0; o < D_output.Length; o++)
                {
                    sum += D_hidden2[h2].weights[o] * deltaOutReal[o];
                }
                delta2Real[h2] = sum;
            }

            decimal[] delta1Real = new decimal[D_hidden1.Length];
            for (int h1 = 0; h1 < D_hidden1.Length; h1++)
            {
                decimal sum = 0m;
                for (int h2 = 0; h2 < D_hidden2.Length; h2++)
                {
                    sum += D_hidden1[h1].weights[h2] * delta2Real[h2];
                }
                delta1Real[h1] = sum * ReluPrimeFromActivation(D_hidden1[h1].x);
            }

            for (int h2 = 0; h2 < D_hidden2.Length; h2++)
            {
                for (int o = 0; o < D_output.Length; o++)
                {
                    D_hidden2[h2].weights[o] = ApplyWeightUpdate(
                        D_hidden2[h2].weights[o],
                        D_hidden2[h2].x,
                        deltaOutReal[o],
                        updateRate
                    );
                }
            }

            for (int h1 = 0; h1 < D_hidden1.Length; h1++)
            {
                for (int h2 = 0; h2 < D_hidden2.Length; h2++)
                {
                    D_hidden1[h1].weights[h2] = ApplyWeightUpdate(
                        D_hidden1[h1].weights[h2],
                        Relu(D_hidden1[h1].x),
                        delta2Real[h2],
                        updateRate
                    );
                }
            }

            for (int inp = 0; inp < D_input.Length; inp++)
            {
                for (int h1 = 0; h1 < D_hidden1.Length; h1++)
                {
                    D_input[inp].weights[h1] = ApplyWeightUpdate(
                        D_input[inp].weights[h1],
                        D_input[inp].x,
                        delta1Real[h1],
                        updateRate
                    );
                }
            }

            // D(fake): 생성 샘플은 fake 클래스(index 10)가 1이 되도록 학습
            for (int i = 0; i < D_input.Length && i < G_output.Length; i++)
            {
                D_input[i].x = G_output[i].x;
            }

            for (int h1 = 0; h1 < D_hidden1.Length; h1++)
            {
                D_hidden1[h1].Calculate(D_input, h1);
            }
            for (int h2 = 0; h2 < D_hidden2.Length; h2++)
            {
                D_hidden2[h2].CalculateReLU(D_hidden1, h2);
            }
            for (int o = 0; o < D_output.Length; o++)
            {
                D_output[o].Calculate(D_hidden2, o);
                D_output[o].x = Sigmoid(D_output[o].x); // sigmoid으로 확률화
            }

            int fakeClassIndex = D_output.Length - 1;
            decimal[] deltaOutFake = new decimal[D_output.Length];
            for (int o = 0; o < D_output.Length; o++)
            {
                decimal target = (o == fakeClassIndex) ? 1m : 0m;
                deltaOutFake[o] = D_output[o].x - target;
            }

            decimal[] delta2Fake = new decimal[D_hidden2.Length];
            for (int h2 = 0; h2 < D_hidden2.Length; h2++)
            {
                decimal sum = 0m;
                for (int o = 0; o < D_output.Length; o++)
                {
                    sum += D_hidden2[h2].weights[o] * deltaOutFake[o];
                }
                delta2Fake[h2] = sum;
            }

            decimal[] delta1Fake = new decimal[D_hidden1.Length];
            for (int h1 = 0; h1 < D_hidden1.Length; h1++)
            {
                decimal sum = 0m;
                for (int h2 = 0; h2 < D_hidden2.Length; h2++)
                {
                    sum += D_hidden1[h1].weights[h2] * delta2Fake[h2];
                }
                delta1Fake[h1] = sum * ReluPrimeFromActivation(D_hidden1[h1].x);
            }

            for (int h2 = 0; h2 < D_hidden2.Length; h2++)
            {
                for (int o = 0; o < D_output.Length; o++)
                {
                    D_hidden2[h2].weights[o] = ApplyWeightUpdate(
                        D_hidden2[h2].weights[o],
                        D_hidden2[h2].x,
                        deltaOutFake[o],
                        updateRate
                    );
                }
            }

            for (int h1 = 0; h1 < D_hidden1.Length; h1++)
            {
                for (int h2 = 0; h2 < D_hidden2.Length; h2++)
                {
                    D_hidden1[h1].weights[h2] = ApplyWeightUpdate(
                        D_hidden1[h1].weights[h2],
                        Relu(D_hidden1[h1].x),
                        delta2Fake[h2],
                        updateRate
                    );
                }
            }

            for (int inp = 0; inp < D_input.Length; inp++)
            {
                for (int h1 = 0; h1 < D_hidden1.Length; h1++)
                {
                    D_input[inp].weights[h1] = ApplyWeightUpdate(
                        D_input[inp].weights[h1],
                        D_input[inp].x,
                        delta1Fake[h1],
                        updateRate
                    );
                }
            }

            // G 업데이트: D(fake)의 그래디언트를 D 입력(=G 출력)까지 전파한 뒤 G 파라미터를 갱신한다.
            for (int i = 0; i < D_input.Length && i < G_output.Length; i++)
            {
                D_input[i].x = G_output[i].x;
            }
            for (int h1 = 0; h1 < D_hidden1.Length; h1++)
            {
                D_hidden1[h1].Calculate(D_input, h1);
            }
            for (int h2 = 0; h2 < D_hidden2.Length; h2++)
            {
                D_hidden2[h2].CalculateReLU(D_hidden1, h2);
            }
            for (int o = 0; o < D_output.Length; o++)
            {
                D_output[o].Calculate(D_hidden2, o);
                D_output[o].x = Sigmoid(D_output[o].x);
            }

            decimal[] dDeltaOutForG = new decimal[D_output.Length];
            for (int o = 0; o < D_output.Length; o++)
            {
                decimal target = (o == generatedDigit) ? 1m : 0m;
                dDeltaOutForG[o] = D_output[o].x - target;
            }

            decimal[] dDelta2ForG = new decimal[D_hidden2.Length];
            for (int h2 = 0; h2 < D_hidden2.Length; h2++)
            {
                decimal sum = 0m;
                for (int o = 0; o < D_output.Length; o++)
                {
                    sum += D_hidden2[h2].weights[o] * dDeltaOutForG[o];
                }
                dDelta2ForG[h2] = sum;
            }

            decimal[] dDelta1ForG = new decimal[D_hidden1.Length];
            for (int h1 = 0; h1 < D_hidden1.Length; h1++)
            {
                decimal sum = 0m;
                for (int h2 = 0; h2 < D_hidden2.Length; h2++)
                {
                    sum += D_hidden1[h1].weights[h2] * dDelta2ForG[h2];
                }
                dDelta1ForG[h1] = sum * ReluPrimeFromActivation(D_hidden1[h1].x);
            }

            int gOutConnected = G_hidden2.Length > 0 ? G_hidden2[0].weights.Length : 0;
            int gOutUsed = Math.Min(Math.Min(G_output.Length, D_input.Length), gOutConnected);
            decimal[] gDeltaOut = new decimal[gOutUsed];
            for (int go = 0; go < gOutUsed; go++)
            {
                decimal sum = 0m;
                for (int h1 = 0; h1 < D_hidden1.Length; h1++)
                {
                    sum += D_input[go].weights[h1] * dDelta1ForG[h1];
                }
                gDeltaOut[go] = sum;
            }

            decimal[] gDelta2 = new decimal[G_hidden2.Length];
            for (int h2 = 0; h2 < G_hidden2.Length; h2++)
            {
                decimal sum = 0m;
                for (int go = 0; go < gOutUsed; go++)
                {
                    sum += G_hidden2[h2].weights[go] * gDeltaOut[go];
                }
                gDelta2[h2] = sum;
            }

            decimal[] gDelta1 = new decimal[G_hidden1.Length];
            for (int h1 = 0; h1 < G_hidden1.Length; h1++)
            {
                decimal sum = 0m;
                for (int h2 = 0; h2 < G_hidden2.Length; h2++)
                {
                    sum += G_hidden1[h1].weights[h2] * gDelta2[h2];
                }
                gDelta1[h1] = sum * ReluPrimeFromActivation(G_hidden1[h1].x);
            }

            for (int h2 = 0; h2 < G_hidden2.Length; h2++)
            {
                for (int go = 0; go < gOutUsed; go++)
                {
                    G_hidden2[h2].weights[go] = ApplyWeightUpdate(
                        G_hidden2[h2].weights[go],
                        Relu(G_hidden2[h2].x),
                        gDeltaOut[go],
                        updateRate
                    );
                }
            }

            for (int h1 = 0; h1 < G_hidden1.Length; h1++)
            {
                for (int h2 = 0; h2 < G_hidden2.Length; h2++)
                {
                    G_hidden1[h1].weights[h2] = ApplyWeightUpdate(
                        G_hidden1[h1].weights[h2],
                        Relu(G_hidden1[h1].x),
                        gDelta2[h2],
                        updateRate
                    );
                }
            }

            for (int inp = 0; inp < G_input.Length; inp++)
            {
                for (int h1 = 0; h1 < G_hidden1.Length; h1++)
                {
                    G_input[inp].weights[h1] = ApplyWeightUpdate(
                        G_input[inp].weights[h1],
                        G_input[inp].x,
                        gDelta1[h1],
                        updateRate
                    );
                }
            }


            Console.WriteLine($"epoch#{epoch + 1} {imageDatas.IndexOf(imageData) + 1}/{imageDatas.Count}");
        }
    }
    {
        string modelPath = Path.Combine(Environment.CurrentDirectory, "model_weights.txt");
        using (var sw = new StreamWriter(modelPath, false))
        {
            sw.WriteLine("# D_input");
            foreach (var n in D_input)
            {
                sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
            }

            sw.WriteLine("# D_hidden1");
            foreach (var n in D_hidden1)
            {
                sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
            }

            sw.WriteLine("# D_hidden2");
            foreach (var n in D_hidden2)
            {
                sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
            }

            sw.WriteLine("# D_output");
            foreach (var n in D_output)
            {
                sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
            }
            sw.WriteLine("# G_input");
            foreach (var n in G_input)
            {
                sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
            }

            sw.WriteLine("# G_hidden1");
            foreach (var n in G_hidden1)
            {
                sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
            }

            sw.WriteLine("# G_hidden2");
            foreach (var n in G_hidden2)
            {
                sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
            }

            sw.WriteLine("# G_output");
            foreach (var n in G_output)
            {
                sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
            }
        }
    }
}

static void GTest()
{
    Console.WriteLine("Enter a digit to generate (0-9):");
    string? rawDigit = Console.ReadLine()?.Trim();
    if (!int.TryParse(rawDigit, out int digit) || digit < 0 || digit > 9)
    {
        Console.WriteLine("Invalid digit. Please enter a value from 0 to 9.");
        return;
    }

    Node[] G_input = new Node[10];
    Node[] G_hidden1 = new Node[128];
    Node[] G_hidden2 = new Node[128];
    Node[] G_output = new Node[28 * 28];

    for (int i = 0; i < G_input.Length; i++) G_input[i] = new Node(new decimal[128]);
    for (int i = 0; i < G_hidden1.Length; i++) G_hidden1[i] = new Node(new decimal[128]);
    for (int i = 0; i < G_hidden2.Length; i++) G_hidden2[i] = new Node(new decimal[28 * 28]);
    for (int i = 0; i < G_output.Length; i++) G_output[i] = new Node();

    string modelPath = Path.Combine(Environment.CurrentDirectory, "model_weights.txt");
    if (!File.Exists(modelPath))
    {
        Console.WriteLine("model_weights.txt not found. Please run 'train' first.");
        return;
    }

    var lines = File.ReadAllLines(modelPath).Select(l => l.Trim()).Where(l => l.Length > 0).ToList();
    string section = "";
    int inpIndex = 0, h1Index = 0, h2Index = 0;
    foreach (var line in lines)
    {
        if (line.StartsWith("#"))
        {
            if (line.Contains("G_input")) section = "G_input";
            else if (line.Contains("G_hidden1")) section = "G_hidden1";
            else if (line.Contains("G_hidden2")) section = "G_hidden2";
            else section = "";
            continue;
        }

        if (section != "G_input" && section != "G_hidden1" && section != "G_hidden2")
        {
            continue;
        }

        var parts = line.Split(',');
        try
        {
            if (section == "G_input" && inpIndex < G_input.Length)
            {
                G_input[inpIndex].weights = parts.Select(p => decimal.Parse(p, CultureInfo.InvariantCulture)).ToArray();
                inpIndex++;
            }
            else if (section == "G_hidden1" && h1Index < G_hidden1.Length)
            {
                G_hidden1[h1Index].weights = parts.Select(p => decimal.Parse(p, CultureInfo.InvariantCulture)).ToArray();
                h1Index++;
            }
            else if (section == "G_hidden2" && h2Index < G_hidden2.Length)
            {
                G_hidden2[h2Index].weights = parts.Select(p => decimal.Parse(p, CultureInfo.InvariantCulture)).ToArray();
                h2Index++;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to parse weights line: {line}. Exception: {ex.Message}");
            return;
        }
    }

    if (inpIndex < G_input.Length || h1Index < G_hidden1.Length || h2Index < G_hidden2.Length)
    {
        Console.WriteLine("G model weights are incomplete in model_weights.txt.");
        return;
    }

    int gOutConnected = G_hidden2.Min(n => n.weights.Length);
    int generatedOutputCount = Math.Min(G_output.Length, gOutConnected);
    if (generatedOutputCount <= 0)
    {
        Console.WriteLine("G model output weights are invalid.");
        return;
    }

    for (int i = 0; i < G_input.Length; i++)
    {
        G_input[i].x = i == digit ? 1m : 0m;
    }

    for (int h1 = 0; h1 < G_hidden1.Length; h1++)
    {
        G_hidden1[h1].Calculate(G_input, h1);
    }

    for (int h2 = 0; h2 < G_hidden2.Length; h2++)
    {
        G_hidden2[h2].CalculateReLU(G_hidden1, h2);
    }

    for (int o = 0; o < generatedOutputCount; o++)
    {
        G_output[o].Calculate(G_hidden2, o);
        G_output[o].x = Sigmoid(G_output[o].x);
    }
    for (int o = generatedOutputCount; o < G_output.Length; o++)
    {
        G_output[o].x = 0m;
    }

    string outputPath = Path.Combine(Environment.CurrentDirectory, $"{digit}.png");
    using (Bitmap bitmap = new Bitmap(28, 28))
    {
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                int index = i * 28 + j;
                decimal foreground = G_output[index].x;
                if (foreground < 0m) foreground = 0m;
                if (foreground > 1m) foreground = 1m;

                int gray = Convert.ToInt32(Math.Round(Convert.ToDouble((1m - foreground) * 255m)));
                gray = Math.Clamp(gray, 0, 255);
                bitmap.SetPixel(i, j, Color.FromArgb(gray, gray, gray));
            }
        }

        bitmap.Save(outputPath);
    }

    Console.WriteLine($"Generated image saved: {outputPath}");
    if (generatedOutputCount < G_output.Length)
    {
        Console.WriteLine($"Warning: G_hidden2 output width is {generatedOutputCount}, so remaining pixels were filled with 0.");
    }
}

static void DTest()
{
    // Prepare nodes structures
    Node[] D_input = new Node[28 * 28];
    Node[] D_hidden1 = new Node[128];
    Node[] D_hidden2 = new Node[128];
    Node[] D_output = new Node[11];

    // Initialize with default arrays to be replaced by loaded weights
    for (int i = 0; i < D_input.Length; i++) D_input[i] = new Node(new decimal[128]);
    for (int i = 0; i < D_hidden1.Length; i++) D_hidden1[i] = new Node(new decimal[128]);
    for (int i = 0; i < D_hidden2.Length; i++) D_hidden2[i] = new Node(new decimal[11]);
    for (int i = 0; i < D_output.Length; i++) D_output[i] = new Node();

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
            if (line.Contains("D_input")) section = "D_input";
            else if (line.Contains("D_hidden1")) section = "D_hidden1";
            else if (line.Contains("D_hidden2")) section = "D_hidden2";
            else if (line.Contains("D_output")) section = "D_output";
            continue;
        }

        var parts = line.Split(',');
        try
        {
            if (section == "D_input" && inpIndex < D_input.Length)
            {
                D_input[inpIndex].weights = parts.Select(p => decimal.Parse(p, CultureInfo.InvariantCulture)).ToArray();
                inpIndex++;
            }
            else if (section == "D_hidden1" && h1Index < D_hidden1.Length)
            {
                D_hidden1[h1Index].weights = parts.Select(p => decimal.Parse(p, CultureInfo.InvariantCulture)).ToArray();
                h1Index++;
            }
            else if (section == "D_hidden2" && h2Index < D_hidden2.Length)
            {
                D_hidden2[h2Index].weights = parts.Select(p => decimal.Parse(p, CultureInfo.InvariantCulture)).ToArray();
                h2Index++;
            }
            else if (section == "D_output" && outIndex < D_output.Length)
            {
                D_output[outIndex].weights = parts.Select(p => decimal.Parse(p, CultureInfo.InvariantCulture)).ToArray();
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
                D_input[i * 28 + j].x = NormalizedPixelValue(pixelColor);
            }
        }

        for (int h1 = 0; h1 < D_hidden1.Length; h1++)
        {
            D_hidden1[h1].Calculate(D_input, h1);
        }

        for (int h2 = 0; h2 < D_hidden2.Length; h2++)
        {
            D_hidden2[h2].CalculateReLU(D_hidden1, h2);
        }

        for (int o = 0; o < D_output.Length; o++)
        {
            D_output[o].Calculate(D_hidden2, o);
            D_output[o].x = Sigmoid(D_output[o].x); // sigmoid으로 확률화
        }

        // prediction
        int pred = 0;
        decimal max = D_output[0].x;
        for (int k = 1; k < 10; k++)
        {
            if (D_output[k].x > max) { max = D_output[k].x; pred = k; }
        }
        if (pred == imageData.Label) correct++;

        Console.WriteLine($"{idxImg + 1}/{total} - label: {imageData.Label}, pred: {pred}");
    }

    Console.WriteLine($"Test complete. Accuracy: {(double)correct / total:P2} ({correct}/{total})");
}
