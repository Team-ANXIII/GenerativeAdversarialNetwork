using System;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using back_propagation;

const int ImageVectorSize = 28 * 28;
const int ConditionVectorSize = 10;
const int HiddenSize = 128;

const decimal UpdateActivationClip = 10m;
const decimal UpdateDeltaClip = 10m;
const decimal UpdateGradientClip = 25m;
const decimal WeightClip = 5m;

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
    return a > 0m ? 1m : 0m;
}

static decimal Relu(decimal x)
{
    return x > 0m ? x : 0m;
}

static decimal Sigmoid(decimal x)
{
    double z = Math.Clamp(Convert.ToDouble(x), -40.0, 40.0);
    return Convert.ToDecimal(1.0 / (1.0 + Math.Exp(-z)));
}

static decimal SigmoidPrimeFromActivation(decimal y)
{
    return y * (1m - y);
}

static decimal Clamp(decimal value, decimal min, decimal max)
{
    if (value < min) return min;
    if (value > max) return max;
    return value;
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
    decimal gray = Convert.ToDecimal(((0.299 * pixelColor.R) + (0.587 * pixelColor.G) + (0.114 * pixelColor.B)) / 255.0);
    return 1m - gray;
}

static decimal XavierUniformWeight(Random rand, int fanIn)
{
    double limit = Math.Sqrt(6.0 / fanIn);
    double sampled = ((rand.NextDouble() * 2.0) - 1.0) * limit;
    return Convert.ToDecimal(sampled);
}

static int ReadPositiveEnvInt(string key)
{
    string? raw = Environment.GetEnvironmentVariable(key);
    if (int.TryParse(raw, out int parsed) && parsed > 0) return parsed;
    return 0;
}

static void SetDiscriminatorCondition(Node[] dInput, int label)
{
    int condStart = ImageVectorSize;
    for (int i = 0; i < ConditionVectorSize; i++)
    {
        dInput[condStart + i].x = i == label ? 1m : 0m;
    }
}

static decimal ForwardDiscriminator(Node[] dInput, Node[] dHidden1, Node[] dHidden2, Node[] dOutput)
{
    for (int h1 = 0; h1 < dHidden1.Length; h1++)
    {
        dHidden1[h1].Calculate(dInput, h1);
    }

    for (int h2 = 0; h2 < dHidden2.Length; h2++)
    {
        dHidden2[h2].CalculateReLU(dHidden1, h2);
    }

    dOutput[0].Calculate(dHidden2, 0);
    dOutput[0].x = Sigmoid(dOutput[0].x);
    return dOutput[0].x;
}

static void ForwardGenerator(Node[] gInput, Node[] gHidden1, Node[] gHidden2, Node[] gOutput)
{
    for (int h1 = 0; h1 < gHidden1.Length; h1++)
    {
        gHidden1[h1].Calculate(gInput, h1);
    }

    for (int h2 = 0; h2 < gHidden2.Length; h2++)
    {
        gHidden2[h2].CalculateReLU(gHidden1, h2);
    }

    for (int o = 0; o < gOutput.Length; o++)
    {
        gOutput[o].Calculate(gHidden2, o);
        gOutput[o].x = Sigmoid(gOutput[o].x);
    }
}

static void Train()
{
    string trainDirPath = Path.Combine(Environment.CurrentDirectory, "mnist_png", "training");
    if (!Directory.Exists(trainDirPath))
    {
        Console.WriteLine($"Training directory not found: {trainDirPath}");
        return;
    }

    var trainDir = new DirectoryInfo(trainDirPath);
    var imageDatas = new List<ImageData>();
    foreach (var subdir in trainDir.GetDirectories())
    {
        Console.WriteLine(subdir.FullName);
        foreach (var file in subdir.GetFiles())
        {
            imageDatas.Add(new ImageData(file.FullName, int.Parse(subdir.Name), int.Parse(file.Name.Split('.')[0])));
        }
    }

    imageDatas.Sort((a, b) => a.FileName.CompareTo(b.FileName));
    int maxTrainSamples = ReadPositiveEnvInt("MAX_TRAIN_SAMPLES");
    if (maxTrainSamples > 0 && maxTrainSamples < imageDatas.Count)
    {
        imageDatas = imageDatas.Take(maxTrainSamples).ToList();
        Console.WriteLine($"Training sample limit enabled: {imageDatas.Count}");
    }

    Node[] dInput = new Node[ImageVectorSize + ConditionVectorSize];
    Node[] dHidden1 = new Node[HiddenSize];
    Node[] dHidden2 = new Node[HiddenSize];
    Node[] dOutput = new Node[1]; // one-dimensional discriminator output (realness)

    Node[] gInput = new Node[ConditionVectorSize];
    Node[] gHidden1 = new Node[HiddenSize];
    Node[] gHidden2 = new Node[HiddenSize];
    Node[] gOutput = new Node[ImageVectorSize];

    Random rand = new Random(DateTime.Now.Millisecond);
    decimal updateRate = 0.0002m;

    for (int i = 0; i < dInput.Length; i++)
    {
        decimal[] w = new decimal[HiddenSize];
        for (int j = 0; j < HiddenSize; j++) w[j] = XavierUniformWeight(rand, dInput.Length);
        dInput[i] = new Node(w);
    }

    for (int i = 0; i < dHidden1.Length; i++)
    {
        decimal[] w = new decimal[HiddenSize];
        for (int j = 0; j < HiddenSize; j++) w[j] = XavierUniformWeight(rand, HiddenSize);
        dHidden1[i] = new Node(w);
    }

    for (int i = 0; i < dHidden2.Length; i++)
    {
        decimal[] w = new decimal[1];
        w[0] = XavierUniformWeight(rand, HiddenSize);
        dHidden2[i] = new Node(w);
    }
    dOutput[0] = new Node();

    for (int i = 0; i < gInput.Length; i++)
    {
        decimal[] w = new decimal[HiddenSize];
        for (int j = 0; j < HiddenSize; j++) w[j] = XavierUniformWeight(rand, gInput.Length);
        gInput[i] = new Node(w);
    }

    for (int i = 0; i < gHidden1.Length; i++)
    {
        decimal[] w = new decimal[HiddenSize];
        for (int j = 0; j < HiddenSize; j++) w[j] = XavierUniformWeight(rand, HiddenSize);
        gHidden1[i] = new Node(w);
    }

    for (int i = 0; i < gHidden2.Length; i++)
    {
        decimal[] w = new decimal[ImageVectorSize];
        for (int j = 0; j < ImageVectorSize; j++) w[j] = XavierUniformWeight(rand, HiddenSize);
        gHidden2[i] = new Node(w);
    }

    for (int i = 0; i < gOutput.Length; i++) gOutput[i] = new Node();

    int epochs = ReadPositiveEnvInt("EPOCHS");
    if (epochs <= 0) epochs = 10;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int i = imageDatas.Count - 1; i > 0; i--)
        {
            int j = rand.Next(i + 1);
            (imageDatas[i], imageDatas[j]) = (imageDatas[j], imageDatas[i]);
        }

        Console.WriteLine($"Epoch {epoch + 1}/{epochs}");
        for (int sampleIndex = 0; sampleIndex < imageDatas.Count; sampleIndex++)
        {
            var imageData = imageDatas[sampleIndex];
            using Bitmap bitmap = new Bitmap(imageData.ImagePath);

            for (int i = 0; i < bitmap.Width; i++)
            {
                for (int j = 0; j < bitmap.Height; j++)
                {
                    dInput[i * 28 + j].x = NormalizedPixelValue(bitmap.GetPixel(i, j));
                }
            }
            SetDiscriminatorCondition(dInput, imageData.Label);

            int generatedDigit = rand.Next(ConditionVectorSize);
            for (int i = 0; i < gInput.Length; i++)
            {
                gInput[i].x = i == generatedDigit ? 1m : 0m;
            }
            ForwardGenerator(gInput, gHidden1, gHidden2, gOutput);

            // D(real): target = 1
            decimal dReal = ForwardDiscriminator(dInput, dHidden1, dHidden2, dOutput);
            decimal deltaOutReal = dReal - 1m;

            decimal[] delta2Real = new decimal[dHidden2.Length];
            for (int h2 = 0; h2 < dHidden2.Length; h2++)
            {
                delta2Real[h2] = dHidden2[h2].weights[0] * deltaOutReal;
            }

            decimal[] delta1Real = new decimal[dHidden1.Length];
            for (int h1 = 0; h1 < dHidden1.Length; h1++)
            {
                decimal sum = 0m;
                for (int h2 = 0; h2 < dHidden2.Length; h2++) sum += dHidden1[h1].weights[h2] * delta2Real[h2];
                delta1Real[h1] = sum * ReluPrimeFromActivation(dHidden1[h1].x);
            }

            for (int h2 = 0; h2 < dHidden2.Length; h2++)
            {
                dHidden2[h2].weights[0] = ApplyWeightUpdate(dHidden2[h2].weights[0], dHidden2[h2].x, deltaOutReal, updateRate);
            }

            for (int h1 = 0; h1 < dHidden1.Length; h1++)
            {
                for (int h2 = 0; h2 < dHidden2.Length; h2++)
                {
                    dHidden1[h1].weights[h2] = ApplyWeightUpdate(dHidden1[h1].weights[h2], Relu(dHidden1[h1].x), delta2Real[h2], updateRate);
                }
            }

            for (int inp = 0; inp < dInput.Length; inp++)
            {
                for (int h1 = 0; h1 < dHidden1.Length; h1++)
                {
                    dInput[inp].weights[h1] = ApplyWeightUpdate(dInput[inp].weights[h1], dInput[inp].x, delta1Real[h1], updateRate);
                }
            }

            // D(fake): target = 0
            for (int i = 0; i < ImageVectorSize; i++) dInput[i].x = gOutput[i].x;
            SetDiscriminatorCondition(dInput, generatedDigit);

            decimal dFake = ForwardDiscriminator(dInput, dHidden1, dHidden2, dOutput);
            decimal deltaOutFake = dFake - 0m;

            decimal[] delta2Fake = new decimal[dHidden2.Length];
            for (int h2 = 0; h2 < dHidden2.Length; h2++)
            {
                delta2Fake[h2] = dHidden2[h2].weights[0] * deltaOutFake;
            }

            decimal[] delta1Fake = new decimal[dHidden1.Length];
            for (int h1 = 0; h1 < dHidden1.Length; h1++)
            {
                decimal sum = 0m;
                for (int h2 = 0; h2 < dHidden2.Length; h2++) sum += dHidden1[h1].weights[h2] * delta2Fake[h2];
                delta1Fake[h1] = sum * ReluPrimeFromActivation(dHidden1[h1].x);
            }

            for (int h2 = 0; h2 < dHidden2.Length; h2++)
            {
                dHidden2[h2].weights[0] = ApplyWeightUpdate(dHidden2[h2].weights[0], dHidden2[h2].x, deltaOutFake, updateRate);
            }

            for (int h1 = 0; h1 < dHidden1.Length; h1++)
            {
                for (int h2 = 0; h2 < dHidden2.Length; h2++)
                {
                    dHidden1[h1].weights[h2] = ApplyWeightUpdate(dHidden1[h1].weights[h2], Relu(dHidden1[h1].x), delta2Fake[h2], updateRate);
                }
            }

            for (int inp = 0; inp < dInput.Length; inp++)
            {
                for (int h1 = 0; h1 < dHidden1.Length; h1++)
                {
                    dInput[inp].weights[h1] = ApplyWeightUpdate(dInput[inp].weights[h1], dInput[inp].x, delta1Fake[h1], updateRate);
                }
            }

            // G step: wants D(fake) -> 1
            for (int i = 0; i < ImageVectorSize; i++) dInput[i].x = gOutput[i].x;
            SetDiscriminatorCondition(dInput, generatedDigit);

            decimal dForG = ForwardDiscriminator(dInput, dHidden1, dHidden2, dOutput);
            decimal deltaOutForG = dForG - 1m;

            decimal[] dDelta2ForG = new decimal[dHidden2.Length];
            for (int h2 = 0; h2 < dHidden2.Length; h2++)
            {
                dDelta2ForG[h2] = dHidden2[h2].weights[0] * deltaOutForG;
            }

            decimal[] dDelta1ForG = new decimal[dHidden1.Length];
            for (int h1 = 0; h1 < dHidden1.Length; h1++)
            {
                decimal sum = 0m;
                for (int h2 = 0; h2 < dHidden2.Length; h2++) sum += dHidden1[h1].weights[h2] * dDelta2ForG[h2];
                dDelta1ForG[h1] = sum * ReluPrimeFromActivation(dHidden1[h1].x);
            }

            decimal[] gDeltaOut = new decimal[gOutput.Length];
            for (int go = 0; go < gOutput.Length; go++)
            {
                decimal sum = 0m;
                for (int h1 = 0; h1 < dHidden1.Length; h1++) sum += dInput[go].weights[h1] * dDelta1ForG[h1];
                gDeltaOut[go] = sum * SigmoidPrimeFromActivation(gOutput[go].x);
            }

            decimal[] gDelta2 = new decimal[gHidden2.Length];
            for (int h2 = 0; h2 < gHidden2.Length; h2++)
            {
                decimal sum = 0m;
                for (int go = 0; go < gOutput.Length; go++) sum += gHidden2[h2].weights[go] * gDeltaOut[go];
                gDelta2[h2] = sum;
            }

            decimal[] gDelta1 = new decimal[gHidden1.Length];
            for (int h1 = 0; h1 < gHidden1.Length; h1++)
            {
                decimal sum = 0m;
                for (int h2 = 0; h2 < gHidden2.Length; h2++) sum += gHidden1[h1].weights[h2] * gDelta2[h2];
                gDelta1[h1] = sum * ReluPrimeFromActivation(gHidden1[h1].x);
            }

            for (int h2 = 0; h2 < gHidden2.Length; h2++)
            {
                for (int go = 0; go < gOutput.Length; go++)
                {
                    gHidden2[h2].weights[go] = ApplyWeightUpdate(gHidden2[h2].weights[go], gHidden2[h2].x, gDeltaOut[go], updateRate);
                }
            }

            for (int h1 = 0; h1 < gHidden1.Length; h1++)
            {
                for (int h2 = 0; h2 < gHidden2.Length; h2++)
                {
                    gHidden1[h1].weights[h2] = ApplyWeightUpdate(gHidden1[h1].weights[h2], Relu(gHidden1[h1].x), gDelta2[h2], updateRate);
                }
            }

            for (int inp = 0; inp < gInput.Length; inp++)
            {
                for (int h1 = 0; h1 < gHidden1.Length; h1++)
                {
                    gInput[inp].weights[h1] = ApplyWeightUpdate(gInput[inp].weights[h1], gInput[inp].x, gDelta1[h1], updateRate);
                }
            }

            Console.WriteLine($"epoch#{epoch + 1} {sampleIndex + 1}/{imageDatas.Count}");
        }

        string modelPath = Path.Combine(Environment.CurrentDirectory, "model_weights.txt");
        using (var sw = new StreamWriter(modelPath, false))
        {
            sw.WriteLine("# D_input");
            foreach (var n in dInput) sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));

            sw.WriteLine("# D_hidden1");
            foreach (var n in dHidden1) sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));

            sw.WriteLine("# D_hidden2");
            foreach (var n in dHidden2) sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));

            sw.WriteLine("# D_output");
            foreach (var n in dOutput) sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));

            sw.WriteLine("# G_input");
            foreach (var n in gInput) sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));

            sw.WriteLine("# G_hidden1");
            foreach (var n in gHidden1) sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));

            sw.WriteLine("# G_hidden2");
            foreach (var n in gHidden2) sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));

            sw.WriteLine("# G_output");
            foreach (var n in gOutput) sw.WriteLine(string.Join(",", n.weights.Select(w => w.ToString(CultureInfo.InvariantCulture))));
        }
    }
}

static void Test()
{
    Console.WriteLine("Enter a digit to generate (0-9):");
    string? rawDigit = Console.ReadLine()?.Trim();
    if (!int.TryParse(rawDigit, out int digit) || digit < 0 || digit > 9)
    {
        Console.WriteLine("Invalid digit. Please enter a value from 0 to 9.");
        return;
    }

    Node[] gInput = new Node[ConditionVectorSize];
    Node[] gHidden1 = new Node[HiddenSize];
    Node[] gHidden2 = new Node[HiddenSize];
    Node[] gOutput = new Node[ImageVectorSize];

    for (int i = 0; i < gInput.Length; i++) gInput[i] = new Node(new decimal[HiddenSize]);
    for (int i = 0; i < gHidden1.Length; i++) gHidden1[i] = new Node(new decimal[HiddenSize]);
    for (int i = 0; i < gHidden2.Length; i++) gHidden2[i] = new Node(new decimal[ImageVectorSize]);
    for (int i = 0; i < gOutput.Length; i++) gOutput[i] = new Node();

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
            section = line.Substring(1).Trim();
            continue;
        }

        var parts = line.Split(',');
        try
        {
            if (section == "G_input" && inpIndex < gInput.Length)
            {
                gInput[inpIndex].weights = parts.Select(p => decimal.Parse(p, CultureInfo.InvariantCulture)).ToArray();
                inpIndex++;
            }
            else if (section == "G_hidden1" && h1Index < gHidden1.Length)
            {
                gHidden1[h1Index].weights = parts.Select(p => decimal.Parse(p, CultureInfo.InvariantCulture)).ToArray();
                h1Index++;
            }
            else if (section == "G_hidden2" && h2Index < gHidden2.Length)
            {
                gHidden2[h2Index].weights = parts.Select(p => decimal.Parse(p, CultureInfo.InvariantCulture)).ToArray();
                h2Index++;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to parse weights line: {line}. Exception: {ex.Message}");
            return;
        }
    }

    if (inpIndex < gInput.Length || h1Index < gHidden1.Length || h2Index < gHidden2.Length)
    {
        Console.WriteLine("G model weights are incomplete in model_weights.txt.");
        return;
    }

    for (int i = 0; i < gInput.Length; i++) gInput[i].x = i == digit ? 1m : 0m;
    ForwardGenerator(gInput, gHidden1, gHidden2, gOutput);

    string outputPath = Path.Combine(Environment.CurrentDirectory, $"{digit}.png");
    using (Bitmap bitmap = new Bitmap(28, 28))
    {
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                int index = i * 28 + j;
                decimal foreground = Clamp(gOutput[index].x, 0m, 1m);
                int gray = Convert.ToInt32(Math.Round(Convert.ToDouble((1m - foreground) * 255m)));
                gray = Math.Clamp(gray, 0, 255);
                bitmap.SetPixel(i, j, Color.FromArgb(gray, gray, gray));
            }
        }
        bitmap.Save(outputPath);
    }

    Console.WriteLine($"Generated image saved: {outputPath}");
}
