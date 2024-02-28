using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace DigitRecognition
{
    public abstract class NeuralNetwork
    {
        protected const int InputNeuronCount = 784;
        protected const int HiddenNeuronCount = 48;
        protected const int OutputNeuronCount = 10;

        protected const float LearningRate = .08f;
        protected const int Epochs = 3;

        public abstract void Train();
        public abstract float Test();

        protected static DigitImage[] GetImages(bool test)
        {
            Console.WriteLine("\nBegin\n");
            FileStream ifsLabels =
             new FileStream(test ? @"E:\Users\Stephen\Applications\MyProgram\DigitRecognition\DigitRecognition\t10k-labels.idx1-ubyte" : @"E:\Users\Stephen\Applications\MyProgram\DigitRecognition\DigitRecognition\train-labels.idx1-ubyte",
             FileMode.Open); // test labels
            FileStream ifsImages =
             new FileStream(test ? @"E:\Users\Stephen\Applications\MyProgram\DigitRecognition\DigitRecognition\t10k-images.idx3-ubyte" : @"E:\Users\Stephen\Applications\MyProgram\DigitRecognition\DigitRecognition\train-images.idx3-ubyte",
             FileMode.Open); // test images

            BinaryReader brLabels =
             new BinaryReader(ifsLabels);
            BinaryReader brImages =
             new BinaryReader(ifsImages);

            brImages.ReadInt32(); // discard
            int numImages = brImages.ReadInt32();
            int numRows = brImages.ReadInt32();
            int numCols = brImages.ReadInt32();

            brLabels.ReadInt32();
            int numLabels = brLabels.ReadInt32();

            byte[][] pixels = new byte[28][];
            for (int i = 0; i < pixels.Length; ++i)
                pixels[i] = new byte[28];

            Console.WriteLine(numImages);
            int num = test ? 9000 : 55000;
            DigitImage[] images = new DigitImage[num];
            // each test image
            for (int di = 0; di < num; ++di)
            {
                for (int i = 0; i < 28; ++i)
                {
                    for (int j = 0; j < 28; ++j)
                    {
                        byte b = brImages.ReadByte();
                        pixels[i][j] = b;
                    }
                }

                byte lbl = brLabels.ReadByte();

                images[di] = new DigitImage(pixels, lbl);
            } // each image

            ifsImages.Close();
            brImages.Close();
            ifsLabels.Close();
            brLabels.Close();

            return images;
        }

        protected float SigmoidDerivative(float x)
        {
            float sig = SigmoidFunction(x);
            return sig * (1 - sig);
        }

        protected float SigmoidFunction(float x)
        {
            return 1 / (1 + MathF.Exp(-x));
        }

        protected float TanHyperbolic(float x)
        {
            return MathF.Tanh(x);
        }

        protected float TanHyperbolicDerivative(float x)
        {
            float tanh = TanHyperbolic(x);
            return 1 - (tanh * tanh);
        }

        protected float ReLu(float x)
        {
            return x < 0 ? 0 : x;
        }

        protected float ReLuDerivative(float x)
        {
            return x < 0 ? 0 : 1;
        }

        protected float ParametricRelu(float x)
        {
            float a = 0.2f;
            return x < 0 ? a * x : x;
        }

        protected float ParametricReluDerivative(float x)
        {
            float a = 0.2f;
            return x < 0 ? a : 1;
        }

        protected float ExpRelu(float x)
        {
            float a = 0.5f;
            return x < 0 ? a * (MathF.Exp(x) - 1) : x;
        }

        protected float ExpReluDerivative(float x)
        {
            float a = 0.5f;
            return x < 0 ? a + ExpRelu(x) : 1;
        }

        protected float SoftPlus(float x)
        {
            return MathF.Log(1 + MathF.Exp(x));
        }

        protected float SoftPlusDerivative(float x)
        {
            return SigmoidFunction(x);
        }
    }
    public class DigitImage
    {
        public byte[][] pixels;
        public byte label;

        public DigitImage(byte[][] pixels,
          byte label)
        {
            this.pixels = new byte[28][];
            for (int i = 0; i < this.pixels.Length; ++i)
                this.pixels[i] = new byte[28];

            for (int i = 0; i < 28; ++i)
                for (int j = 0; j < 28; ++j)
                    this.pixels[i][j] = pixels[i][j];

            this.label = label;
        }

        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    if (this.pixels[i][j] == 0)
                        s += " "; // white
                    else if (this.pixels[i][j] == 255)
                        s += "O"; // black
                    else
                        s += "."; // gray
                }
                s += "\n";
            }
            s += this.label.ToString();
            return s;
        } // ToString

    }
}
