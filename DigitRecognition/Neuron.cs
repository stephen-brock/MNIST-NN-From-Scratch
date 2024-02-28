using System;
using System.Collections.Generic;
using System.Text;

namespace DigitRecognition
{
    public class NeuronNetwork : NeuralNetwork
    {

        InputNeuron[] inputs = new InputNeuron[InputNeuronCount];
        List<Neuron[]> layers = new List<Neuron[]>()
        {
            new Neuron[HiddenNeuronCount],
            new Neuron[HiddenNeuronCount],
            new Neuron[OutputNeuronCount],
        };

        public NeuronNetwork()
        {

            for (int i = 0; i < InputNeuronCount; i++)
            {
                inputs[i] = new InputNeuron();
            }

            for (int i = 0; i < HiddenNeuronCount; i++)
            {
                layers[0][i] = new Neuron(inputs);
                layers[1][i] = new Neuron(layers[0]);
            }

            for (int i = 0; i < OutputNeuronCount; i++)
            {
                layers[2][i] = new Neuron(layers[1]);
            }

        }

        public override void Train()
        {
            DigitImage[] images = GetImages(false);

            Console.WriteLine(images.Length);


            for (int i = 0; i < images.Length; i++)
            {
                if (i % 25 == 0)
                {
                    Console.WriteLine(i);

                }
                for (int j = 0; j < InputNeuronCount; j++)
                {
                    //Set inputs
                    for (int x = 0; x < 28; x++)
                    {
                        for (int y = 0; y < 28; y++)
                        {
                            inputs[j].SetActivation(images[i].pixels[x][y] / 255f);
                        }
                    }
                }

                foreach (Neuron[] layer in layers)
                {
                    for (int j = 0; j < layer.Length; j++)
                    {
                        layer[j].CalculateActivation();
                    }
                }

                float[] desiredOutput = new float[OutputNeuronCount];
                desiredOutput[images[i].label] = 1;

                float maxCost = 0;

                for (int j = 0; j < OutputNeuronCount; j++)
                {
                    float act = layers[2][j].Activation;
                    float cost = act - desiredOutput[j];
                    float sqrCost = cost * cost;
                    maxCost += sqrCost;

                    layers[2][j].Decent(2 * cost, 5f);
                }
            }
        }

        public override float Test()
        {
            Console.WriteLine("Test");

            DigitImage[] testImages = GetImages(true);

            int successes = 0;

            float avgCost = 0;

            for (int i = 0; i < 500; i++)
            {
                if (i % 25 == 0)
                {
                    Console.WriteLine(i);

                }

                for (int j = 0; j < InputNeuronCount; j++)
                {
                    //Set inputs
                    for (int x = 0; x < 28; x++)
                    {
                        for (int y = 0; y < 28; y++)
                        {
                            inputs[j].SetActivation(testImages[i].pixels[x][y] / 255f);
                        }
                    }
                }

                foreach (Neuron[] layer in layers)
                {
                    for (int j = 0; j < layer.Length; j++)
                    {
                        layer[j].CalculateActivation();
                    }
                }

                float[] desiredOutput = new float[OutputNeuronCount];
                desiredOutput[testImages[i].label] = 1;

                //float[] output = new float[OutputLayerCount];

                float maxCost = 0;

                int highestIndex = -1;
                float highestAct = -1;

                for (int j = 0; j < OutputNeuronCount; j++)
                {
                    float act = layers[2][j].Activation;
                    //output[j] = act;
                    float cost = act - desiredOutput[j];
                    float sqrCost = MathF.Pow(act - desiredOutput[j], 2);
                    maxCost += sqrCost;

                    if (act > highestAct)
                    {
                        highestIndex = j;
                    }
                }

                if (desiredOutput[highestIndex] == 1)
                {
                    successes++;
                }

                avgCost += maxCost;
            }

            avgCost /= testImages.Length;

            return 0;
        }
    }


    public class Neuron : BaseNeuron
    {
        private float bias;
        private BaseNeuron[] inputs;
        private float[] inputWeights;
        private float weightedSum;

        public Neuron(BaseNeuron[] inputs)
        {
            this.inputs = inputs;
            Random rand = new Random();
            bias = (float)(rand.NextDouble() * 2) - 1;
            inputWeights = new float[inputs.Length];
            for (int i = 0; i < inputWeights.Length; i++)
            {
                inputWeights[i] = (float)(rand.NextDouble() * 2) - 1;
            }
        }

        public void CalculateActivation()
        {
            weightedSum = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                weightedSum += inputs[i].Activation * inputWeights[i];
            }

            weightedSum += bias;

            Activation = SigmoidFunction(weightedSum);
        }
        
        public void Decent(float costDiv, float learningRate)
        {
            float sigDiv = SigmoidDerivitive(weightedSum);
            for (int i = 0; i < inputs.Length; i++)
            {
                float changeInWeight = inputs[i].Activation * sigDiv * costDiv;
                inputWeights[i] -= changeInWeight * learningRate;

                if (inputs[i] is Neuron)
                {
                    ((Neuron)inputs[i]).Decent(costDiv, learningRate);
                }
            }

            float changeInBias = sigDiv * costDiv;
            bias -= changeInBias * learningRate;
        }

        private float SigmoidDerivitive(float x)
        {
            float sigX = SigmoidFunction(x);
            return sigX * (1 - sigX);
        }

        private float SigmoidFunction(float x)
        {
            return 1 / (1 + MathF.Pow(MathF.E, -x));
        }
    }


    public class InputNeuron : BaseNeuron
    {
        public void SetActivation(float a)
        {
            Activation = a;
        }
    }

    public abstract class BaseNeuron
    {
        public float Activation
        {
            get => activation;
            protected set => activation = value;
        }
        private float activation;
    }

}
