using System;
using System.Collections.Generic;
using System.Text;
using System.Drawing;
using System.Windows;

namespace DigitRecognition
{
    public class AdjustableNetwork : NeuralNetwork
    {
        public float[][] biases;

        public float[][,] weights;

        public float[][] activations;
        public float[][] weightedSums;

        public float[][] activationDiv;

        private int numberOfHidden;

        private float learningRate;

        public AdjustableNetwork(int[] hiddenLayers, float learningRate)
        {
            this.learningRate = learningRate;

            Random random = new Random();

            numberOfHidden = hiddenLayers.Length;

            biases = new float[numberOfHidden + 1][];
            weightedSums = new float[numberOfHidden + 1][];
            activationDiv = new float[numberOfHidden + 1][];
            for (int i = 0; i < numberOfHidden + 1; i++)
            {
                int amount = i == numberOfHidden ? OutputNeuronCount : hiddenLayers[i];
                biases[i] = new float[amount];
                weightedSums[i] = new float[amount];
                activationDiv[i] = new float[amount];
            }

            activations = new float[numberOfHidden + 2][];
            activations[0] = new float[InputNeuronCount];
            activations[activations.Length - 1] = new float[OutputNeuronCount];
            for (int i = 0; i < numberOfHidden; i++)
            {
                activations[i + 1] = new float[hiddenLayers[i]];
            }

            weights = new float[numberOfHidden + 1][,];

            for (int i = 0; i < numberOfHidden + 1; i++)
            {
                int beforeAmount = i == 0 ? InputNeuronCount : hiddenLayers[i - 1];
                int afterAmount = i == numberOfHidden ? OutputNeuronCount : hiddenLayers[i];

                weights[i] = new float[beforeAmount, afterAmount];

                for (int x = 0; x < beforeAmount; x++)
                {
                    for (int y = 0; y < afterAmount; y++)
                    {
                        weights[i][x, y] = (float)(random.NextDouble() - 0.5f) * 2f;
                    }
                }
            }
        }

        public override void Train()
        {
            Console.WriteLine("Train!");
            DigitImage[] images = GetImages(false);

            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                for (int iter = 0; iter < images.Length; iter++)
                {
                    int res = images[iter].pixels.GetLength(0);
                    for (int i = 0; i < res; i++)
                    {
                        for (int j = 0; j < res; j++)
                        {
                            activations[0][i + (j * res)] = images[iter].pixels[i][j] == 0 ? 0 : 1;
                        }
                    }

                    int desiredOutput = (int)images[iter].label;

                    for (int i = 1; i < activations.Length -1; i++)
                    {
                        for (int x = 0; x < activations[i].Length; x++)
                        {
                            weightedSums[i - 1][x] = 0;

                            for (int y = 0; y < activations[i - 1].Length; y++)
                            {
                                weightedSums[i - 1][x] += activations[i - 1][y] * weights[i - 1][y, x];
                            }

                            weightedSums[i - 1][x] += biases[i - 1][x];
                            activations[i][x] = TanHyperbolic(weightedSums[i - 1][x]);
                        }
                    }


                    for (int i = 0; i < OutputNeuronCount; i++)
                    {
                        int bromanmoment = activations.Length - 2;
                        weightedSums[bromanmoment][i] = 0;

                        for (int y = 0; y < activations[bromanmoment].Length; y++)
                        {
                            weightedSums[bromanmoment][i] += activations[bromanmoment][y] * weights[bromanmoment][y, i];
                        }

                        weightedSums[bromanmoment][i] += biases[bromanmoment][i];
                        activations[activations.Length - 1][i] = SigmoidFunction(weightedSums[bromanmoment][i]);
                    }

                    int lastIndex = activationDiv.Length - 1;
                    for (int i = 0; i < OutputNeuronCount; i++)
                    {
                        int value = i == desiredOutput ? 1 : 0;
                        activationDiv[lastIndex][i] = 2 * (activations[activations.Length - 1][i] - value) * SigmoidDerivative(weightedSums[lastIndex][i]);
                        biases[lastIndex][i] -= activationDiv[lastIndex][i] * learningRate;
                    }

                    for (int i = weights.Length - 1; i > 0; i--)
                    {
                        for (int prevIndex = 0; prevIndex < activationDiv[i - 1].Length; prevIndex++)
                        {
                            activationDiv[i - 1][prevIndex] = 0;
                            for (int nextIndex = 0; nextIndex < activationDiv[i].Length; nextIndex++)
                            {
                                activationDiv[i - 1][prevIndex] += activationDiv[i][nextIndex] * weights[i][prevIndex, nextIndex] * TanHyperbolicDerivative(weightedSums[i - 1][prevIndex]); //Used to be SigmoidDerivative(x)
                                weights[i][prevIndex, nextIndex] -= activationDiv[i][nextIndex] * activations[i][prevIndex] * learningRate;
                            }

                            biases[i - 1][prevIndex] -= activationDiv[i - 1][prevIndex] * learningRate;
                        }
                    }

                    for (int i = 0; i < InputNeuronCount; i++)
                    {
                        for (int j = 0; j < activationDiv[0].Length; j++)
                        {
                            weights[0][i, j] -= activationDiv[0][j] * activations[0][i] * learningRate;
                        }
                    }
                }
            }
        }

        public override float Test()
        {
            int successes = 0;
            DigitImage[] images = GetImages(true);

            for (int iter = 0; iter < images.Length; iter++)
            {
                int res = images[iter].pixels.GetLength(0);
                for (int i = 0; i < res; i++)
                {
                    for (int j = 0; j < res; j++)
                    {
                        activations[0][i + (j * res)] = images[iter].pixels[i][j] == 0 ? 0 : 1;
                    }
                }

                int desiredOutput = (int)images[iter].label;

                for (int i = 1; i < activations.Length - 1; i++)
                {
                    for (int x = 0; x < activations[i].Length; x++)
                    {
                        weightedSums[i - 1][x] = 0;

                        for (int y = 0; y < activations[i - 1].Length; y++)
                        {
                            weightedSums[i - 1][x] += activations[i - 1][y] * weights[i - 1][y, x];
                        }

                        weightedSums[i - 1][x] += biases[i - 1][x];
                        activations[i][x] = TanHyperbolic(weightedSums[i - 1][x]);
                    }
                }

                for (int i = 0; i < OutputNeuronCount; i++)
                {
                    int bromanmoment = activations.Length - 2;
                    weightedSums[bromanmoment][i] = 0;

                    for (int y = 0; y < activations[bromanmoment].Length; y++)
                    {
                        weightedSums[bromanmoment][i] += activations[bromanmoment][y] * weights[bromanmoment][y, i];
                    }

                    weightedSums[bromanmoment][i] += biases[bromanmoment][i];
                    activations[activations.Length - 1][i] = SigmoidFunction(weightedSums[bromanmoment][i]);
                }

                float maxValue = float.MinValue;
                int value = -1;
                
                int lastIndex = activations.Length - 1;
                for (int i = 0; i < OutputNeuronCount; i++)
                {
                    if (activations[lastIndex][i] > maxValue)
                    {
                        value = i;
                        maxValue = activations[lastIndex][i];
                    }
                }

                if (value == desiredOutput)
                {
                    successes++;
                }
            }

            Console.WriteLine("SUCCESSES " + ((float)successes / images.Length));

            return (float)successes / images.Length;
        }
    }



    public class MatrixNetworkImages : NeuralNetwork
    {
        //Biases
        public float[] hiddenBiases;
        public float[] outputBiases;

        //Weights, first index is from and second index is to
        public float[,] inputToHiddenWeights;
        public float[,] hiddenToOutputWeights;

        //Activation of nodes
        private float[] input;
        private float[] hidden;
        private float[] output;
        private float[] hiddenSum;
        private float[] outputSum;

        public MatrixNetworkImages()
        {
            Random random = new Random();

            //Activations
            input = new float[InputNeuronCount];
            hidden = new float[HiddenNeuronCount];
            output = new float[OutputNeuronCount];

            //Assign random values between -1 and 1
            hiddenBiases = new float[HiddenNeuronCount];
            for (int i = 0; i < HiddenNeuronCount; i++)
            {
                //(float)(random.NextDouble() - 0.5f) * 2f
                hiddenBiases[i] = (float)(random.NextDouble() - 0.5f) * 2f;
            }

            outputBiases = new float[OutputNeuronCount];
            for (int i = 0; i < OutputNeuronCount; i++)
            {
                outputBiases[i] = (float)(random.NextDouble() - 0.5f) * 2f;
            }

            inputToHiddenWeights = new float[InputNeuronCount, HiddenNeuronCount];
            for (int i = 0; i < InputNeuronCount; i++)
            {
                for (int j = 0; j < HiddenNeuronCount; j++)
                {
                    inputToHiddenWeights[i, j] = (float)(random.NextDouble() - 0.5f) * 2f;
                }
            }

            hiddenToOutputWeights = new float[HiddenNeuronCount, OutputNeuronCount];
            for (int i = 0; i < HiddenNeuronCount; i++)
            {
                for (int j = 0; j < OutputNeuronCount; j++)
                {
                    hiddenToOutputWeights[i, j] = (float)(random.NextDouble() - 0.5f) * 2f;
                }
            }

            hiddenSum = new float[HiddenNeuronCount];
            outputSum = new float[OutputNeuronCount];
        }

        public override void Train()
        {
            Console.WriteLine("Train");

            DigitImage[] images = GetImages(false);

            float[] hiddenActDiv = new float[HiddenNeuronCount];
            float[] outputActDiv = new float[OutputNeuronCount];

            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                for (int iter = 0; iter < images.Length; iter++)
                {
                    int res = images[iter].pixels.GetLength(0);
                    for (int i = 0; i < res; i++)
                    {
                        for (int j = 0; j < res; j++)
                        {
                            input[i + (j * res)] = images[iter].pixels[i][j] == 0 ? 0 : 1;
                        }
                    }

                    int desiredOutput = (int)images[iter].label;

                    for (int i = 0; i < HiddenNeuronCount; i++)
                    {
                        hiddenSum[i] = 0;
                        for (int j = 0; j < InputNeuronCount; j++)
                        {
                            hiddenSum[i] += input[j] * inputToHiddenWeights[j, i];
                        }
                        hiddenSum[i] += hiddenBiases[i];

                        hidden[i] = SigmoidFunction(hiddenSum[i]); //Used to be SigmoidFunction(x)
                    }


                    for (int i = 0; i < OutputNeuronCount; i++)
                    {
                        outputSum[i] = 0f;
                        for (int j = 0; j < HiddenNeuronCount; j++)
                        {
                            outputSum[i] += hidden[j] * hiddenToOutputWeights[j, i];
                        }
                        outputSum[i] += outputBiases[i];
                        output[i] = SigmoidFunction(outputSum[i]); //Used to be SigmoidFunction(x)
                    }

                    for (int i = 0; i < OutputNeuronCount; i++)
                    {
                        int value = i == desiredOutput ? 1 : 0;
                        outputActDiv[i] = 2 * (output[i] - value) * SigmoidDerivative(outputSum[i]); //Used to be SigmoidDerivative(x)
                        outputBiases[i] -= outputActDiv[i] * LearningRate;
                    }


                    for (int i = 0; i < HiddenNeuronCount; i++)
                    {
                        hiddenActDiv[i] = 0;
                        for (int j = 0; j < OutputNeuronCount; j++)
                        {
                            hiddenActDiv[i] += outputActDiv[j] * hiddenToOutputWeights[i, j] * SigmoidDerivative(hiddenSum[i]); //Used to be SigmoidDerivative(x)
                            hiddenToOutputWeights[i, j] -= outputActDiv[j] * hidden[i] * LearningRate; 
                        }

                        hiddenBiases[i] -= hiddenActDiv[i] * LearningRate;
                    }

                    for (int i = 0; i < InputNeuronCount; i++)
                    {
                        for (int j = 0; j < HiddenNeuronCount; j++)
                        {
                            inputToHiddenWeights[i, j] -= hiddenActDiv[j] * input[i] * LearningRate;
                        }
                    }

                }
            
            }
        }

        public override float Test()
        {
            int successes = 0;
            DigitImage[] images = GetImages(true);

            for (int iter = 0; iter < images.Length; iter++)
            {
                int res = images[iter].pixels.GetLength(0);
                for (int i = 0; i < res; i++)
                {
                    for (int j = 0; j < res; j++)
                    {
                        input[i + (j * res)] = images[iter].pixels[i][j] == 0 ? 0 : 1;
                    }
                }

                int desiredOutput = (int)images[iter].label;

                for (int i = 0; i < HiddenNeuronCount; i++)
                {
                    hiddenSum[i] = 0;
                    for (int j = 0; j < InputNeuronCount; j++)
                    {
                        hiddenSum[i] += input[j] * inputToHiddenWeights[j, i];
                    }
                    hiddenSum[i] += hiddenBiases[i];

                    hidden[i] = SigmoidFunction(hiddenSum[i]); //Used to be SigmoidFunction(x)
                }


                for (int i = 0; i < OutputNeuronCount; i++)
                {
                    outputSum[i] = 0f;
                    for (int j = 0; j < HiddenNeuronCount; j++)
                    {
                        outputSum[i] += hidden[j] * hiddenToOutputWeights[j, i];
                    }
                    outputSum[i] += outputBiases[i];
                    output[i] = SigmoidFunction(outputSum[i]); //Used to be SigmoidFunction(x)
                }

                float maxValue = float.MinValue;
                int value = -1;

                for (int i = 0; i < OutputNeuronCount; i++)
                {
                    if (output[i] > maxValue)
                    {
                        value = i;
                        maxValue = output[i];
                    }
                }

                if (value == desiredOutput)
                {
                    successes++;
                }
            }


            Console.WriteLine("SUCCESSES " + ((float)successes / images.Length));
            return 0;
        }
    }

    public class MatrixNetwork : NeuralNetwork
    {
        //Biases
        private float[] hiddenBiases;
        private float[] outputBiases;

        //Weights, first index is from and second index is to
        private float[,] inputToHiddenWeights;
        private float[,] hiddenToOutputWeights;

        //Activation of nodes
        private float[] input;
        private float[] hidden;
        private float[] output;

        //Weighted sum for hidden and output nodes
        private float[] hiddenSum;
        private float[] outputSum;

        //Cost function differentated in relation to activation
        private float[] outputCostDiv;
        private float[] hiddenCostDiv;

        private float[] outputDeltaSum;
        private float[] hiddenDeltaSum;

        public MatrixNetwork()
        {
            Random random = new Random();

            //Activations
            input = new float[InputNeuronCount];
            hidden = new float[HiddenNeuronCount];
            output = new float[OutputNeuronCount];

            //Assign random values between -1 and 1
            hiddenBiases = new float[HiddenNeuronCount];
            for (int i = 0; i < HiddenNeuronCount; i++)
            {
                hiddenBiases[i] = (float)(random.NextDouble() - 0.625f) * 1.25f;
            }

            outputBiases = new float[OutputNeuronCount];
            for (int i = 0; i < OutputNeuronCount; i++)
            {
                outputBiases[i] = (float)(random.NextDouble() - 0.625f) * 1.25f;
            }

            inputToHiddenWeights = new float[InputNeuronCount, HiddenNeuronCount];
            for (int i = 0; i < InputNeuronCount; i++)
            {
                for (int j = 0; j < HiddenNeuronCount; j++)
                {
                    inputToHiddenWeights[i,j] = (float)(random.NextDouble() - 0.625f) * 1.25f;
                }
            }

            hiddenToOutputWeights = new float[HiddenNeuronCount, OutputNeuronCount];
            for (int i = 0; i < HiddenNeuronCount; i++)
            {
                for (int j = 0; j < OutputNeuronCount; j++)
                {
                    hiddenToOutputWeights[i, j] = (float)(random.NextDouble() - 0.625f) * 1.25f;
                }
            }


            outputCostDiv = new float[OutputNeuronCount];
            hiddenCostDiv = new float[HiddenNeuronCount];

            hiddenSum = new float[HiddenNeuronCount];
            outputSum = new float[OutputNeuronCount];

            outputDeltaSum = new float[OutputNeuronCount];
            hiddenDeltaSum = new float[HiddenNeuronCount];
        }

        public override void Train()
        {
            DigitImage[] images = GetImages(false);
            int index = 0;
            Console.WriteLine("Train");


            for (int epoch = 0; epoch < 4; epoch++)
            {
                foreach (DigitImage image in images)
                {
                    //Assigns the input nodes a float value depending on image pixels
                    int res = image.pixels.GetLength(0);
                    for (int i = 0; i < res; i++)
                    {
                        for (int j = 0; j < res; j++)
                        {
                            input[i + (res * j)] = image.pixels[i][j] == 0 ? 0 : 1;
                        }
                    }

                    //Calculates activation for each hidden neuron
                    for (int i = 0; i < HiddenNeuronCount; i++)
                    {
                        float activation = 0f;
                        for (int j = 0; j < InputNeuronCount; j++)
                        {
                            activation += inputToHiddenWeights[j, i] * input[j];
                        }

                       // activation += hiddenBiases[i];

                        //Weighted sum
                        hiddenSum[i] = activation;
                        //Activation
                        hidden[i] = SigmoidFunction(activation);
                    }

                    for (int i = 0; i < OutputNeuronCount; i++)
                    {
                        float activation = 0f;
                        for (int j = 0; j < HiddenNeuronCount; j++)
                        {
                            activation += hiddenToOutputWeights[j, i] * hidden[j];
                        }

                        //activation += outputBiases[i];

                        //Weighted sum
                        outputSum[i] = activation;
                        //Activation
                        output[i] = SigmoidFunction(activation);
                    }

                    //Total cost
                    float maxCost = 0f;

                    for (int i = 0; i < OutputNeuronCount; i++)
                    {
                        //If index is the label, that neuron should be 1
                        //All other neurons should be 0
                        //Cost function derivitive * activation derivitive
                        //da/dz * dC/da
                        float cost = output[i] - (i == image.label ? 1 : 0);
                        outputCostDiv[i] = 2 * cost * SigmoidDerivative(outputSum[i]);

                        //Square difference cost function
                        maxCost += cost * cost;
                    }


                    for (int i = 0; i < OutputNeuronCount; i++)
                    {
                        for (int j = 0; j < HiddenNeuronCount; j++)
                        {
                            //Hidden cost div is the sum of the output cost div * each weight
                            hiddenCostDiv[j] += outputCostDiv[i] * hiddenToOutputWeights[j, i];
                            outputDeltaSum[i] += outputCostDiv[i] * hiddenToOutputWeights[j, i];

                            //Weight change is cost div * previous activation
                            //hiddenToOutputWeights[j, i] -= outputCostDiv[i] * output[i] * LearningRate;
                        }

                        //Bias is just cost div
                        //outputBiases[i] -= outputCostDiv[i] * LearningRate;
                    }


                    for (int i = 0; i < HiddenNeuronCount; i++)
                    {
                        //Multiplying by da/dz (activation derivitive)
                        hiddenCostDiv[i] = hiddenCostDiv[i] * SigmoidDerivative(hiddenSum[i]);

                        for (int j = 0; j < InputNeuronCount; j++)
                        {
                            //inputToHiddenWeights[j, i] -= hiddenCostDiv[i] * hidden[i] * LearningRate;
                            hiddenDeltaSum[i] += hiddenCostDiv[i] * inputToHiddenWeights[j, i];
                        }

                        //hiddenBiases[i] -= hiddenCostDiv[i] * LearningRate;
                        hiddenCostDiv[i] = 0;
                    }


                    if (++index % 50 == 0)
                    {
                        for (int i = 0; i < OutputNeuronCount; i++)
                        {
                            float outputDelta = outputDeltaSum[i] / 50f;
                            outputDeltaSum[i] = 0;
                            for (int j = 0; j < HiddenNeuronCount; j++)
                            {
                                //Weight change is cost div * previous activation
                                hiddenToOutputWeights[j, i] -= outputDelta * output[i] * LearningRate;
                            }

                            //Bias is just cost div
                            //outputBiases[i] -= outputDelta * LearningRate;
                        }


                        for (int i = 0; i < HiddenNeuronCount; i++)
                        {
                            float hiddenDelta = hiddenDeltaSum[i] / 50f;
                            hiddenDeltaSum[i] = 0;

                            for (int j = 0; j < InputNeuronCount; j++)
                            {
                                inputToHiddenWeights[j, i] -= hiddenDelta * hidden[i] * LearningRate;
                            }

                            hiddenBiases[i] -= hiddenDelta * LearningRate;
                            //hiddenCostDiv[i] = 0;
                        }
                    }
                }
            }
        }

        public override float Test()
        {
            DigitImage[] images = GetImages(true);

            Console.WriteLine("Test");

            int successes = 0;
            foreach (DigitImage image in images)
            {

                int res = image.pixels.GetLength(0);
                for (int i = 0; i < res; i++)
                {
                    for (int j = 0; j < res; j++)
                    {
                        input[i + (res * j)] = image.pixels[i][j];
                    }
                }

                for (int i = 0; i < HiddenNeuronCount; i++)
                {
                    float activation = 0f;
                    for (int j = 0; j < InputNeuronCount; j++)
                    {
                        activation += inputToHiddenWeights[j, i] * input[j];
                    }

                   // activation += hiddenBiases[i];

                    hidden[i] = SigmoidFunction(activation);
                }

                float max = float.MinValue;
                int maxIndex = -1;

                for (int i = 0; i < OutputNeuronCount; i++)
                {
                    float activation = 0f;
                    for (int j = 0; j < HiddenNeuronCount; j++)
                    {
                        activation += hiddenToOutputWeights[j, i] * hidden[j];
                    }

                    //activation += outputBiases[i];

                    output[i] = SigmoidFunction(activation);

                    if (output[i] > max)
                    {
                        maxIndex = i;
                        max = output[i];
                    }
                }

                if (image.label == maxIndex)
                {
                    successes++;
                }
            }

            Console.WriteLine(successes / images.Length);
            Console.ReadLine();

            return 0;
        }

        public int GuessImage(float[,] image)
        {
            int res = image.GetLength(0);
            for (int i = 0; i < res; i++)
            {
                for (int j = 0; j < res; j++)
                {
                    input[i + (res * j)] = image[i,j];
                }
            }

            for (int i = 0; i < HiddenNeuronCount; i++)
            {
                float activation = 0f;
                for (int j = 0; j < InputNeuronCount; j++)
                {
                    activation += inputToHiddenWeights[j, i] * input[j];
                }

                // activation += hiddenBiases[i];

                hidden[i] = SigmoidFunction(activation);
            }

            float max = float.MinValue;
            int maxIndex = -1;

            for (int i = 0; i < OutputNeuronCount; i++)
            {
                float activation = 0f;
                for (int j = 0; j < HiddenNeuronCount; j++)
                {
                    activation += hiddenToOutputWeights[j, i] * hidden[j];
                }

                //activation += outputBiases[i];

                output[i] = SigmoidFunction(activation);

                if (output[i] > max)
                {
                    maxIndex = i;
                    max = output[i];
                }
            }

            return maxIndex;
        }
    }
    

    public class MatrixNetworkXOR : NeuralNetwork
    {
        //Biases
        private float[] hiddenBiases;
        private float outputBias;

        //Weights, first index is from and second index is to
        private float[,] inputToHiddenWeights;
        private float[,] hiddenToOutputWeights;

        //Activation of nodes
        private float[] input;
        private float[] hidden;
        private float[] hiddenSum;
        private float output;
        private float outputSum;


        public MatrixNetworkXOR()
        {
            Random random = new Random();

            //Activations
            input = new float[2];
            hidden = new float[2];

            //Assign random values between -1 and 1
            hiddenBiases = new float[2];
            for (int i = 0; i < 2; i++)
            {
                hiddenBiases[i] = (float)(random.NextDouble() - 0.5f) * 2f;
            }

            inputToHiddenWeights = new float[2, 2];
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    inputToHiddenWeights[i, j] = (float)(random.NextDouble() - 0.5f) * 2f;
                }
            }

            hiddenToOutputWeights = new float[2, 1];
            for (int i = 0; i < 2; i++)
            {
                hiddenToOutputWeights[i, 0] = (float)(random.NextDouble() - 0.5f) * 2f;
            }

            hiddenSum = new float[2];
        }

        public override void Train()
        {
            Random random = new Random();

            Console.WriteLine("Train");


            float[] hiddenActDiv = new float[2];

            for (int iter = 0; iter < 500000; iter++)
            {
                input[0] = random.Next(2);
                input[1] = random.Next(2);

                float desiredOutput = ((input[0] == 1) ^ (input[1] == 1)) ? 1 : 0;

                for (int i = 0; i < 2; i++)
                {
                    hiddenSum[i] = 0;
                    for (int j = 0; j < 2; j++)
                    {
                        hiddenSum[i] += input[j] * inputToHiddenWeights[j, i];
                    }
                    hiddenSum[i] += hiddenBiases[i];

                    hidden[i] = SigmoidFunction(hiddenSum[i]);
                }

                outputSum = 0f;

                for (int i = 0; i < 2; i++)
                {
                    outputSum += hidden[i] * hiddenToOutputWeights[i, 0];
                }
                outputSum += outputBias;

                output = SigmoidFunction(outputSum);

                float outputDiv = 2 * (output - desiredOutput) * SigmoidDerivative(outputSum);
                outputBias -= outputDiv * LearningRate;

                for (int i = 0; i < 2; i++)
                {
                    hiddenActDiv[i] = outputDiv * hiddenToOutputWeights[i, 0] * SigmoidDerivative(hiddenSum[i]);
                    hiddenBiases[i] -= hiddenActDiv[i] * LearningRate;
                    hiddenToOutputWeights[i, 0] -= outputDiv * hidden[i] * LearningRate; //Maybe output instead of hiddedn
                }

                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        inputToHiddenWeights[i, j] -= hiddenActDiv[j] * input[i] * LearningRate;
                    }
                }

            }
        }

        public override float Test()
        {
            Random random = new Random();

            int successes = 0;
            for (int iter = 0; iter < 250; iter++)
            {
                input[0] = random.Next(2);
                input[1] = random.Next(2);

                int desiredOutput = ((input[0] == 1) ^ (input[1] == 1)) ? 1 : 0;

                for (int i = 0; i < 2; i++)
                {
                    hiddenSum[i] = 0;
                    for (int j = 0; j < 2; j++)
                    {
                        hiddenSum[i] += input[j] * inputToHiddenWeights[j, i];
                    }
                    hiddenSum[i] += hiddenBiases[i];

                    hidden[i] = SigmoidFunction(hiddenSum[i]);
                }

                outputSum = 0f;

                for (int i = 0; i < 2; i++)
                {
                    outputSum += hidden[i] * hiddenToOutputWeights[i, 0];
                }
                outputSum += outputBias;

                output = SigmoidFunction(outputSum);

                int value = (int)MathF.Round(output);
                if (value == desiredOutput)
                {
                    successes++;
                }

                Console.WriteLine(string.Format("{0} XOR {1} = {2} \n Guess: {3} Rounded: {4}", input[0], input[1], desiredOutput, output, value));
            }

            Console.WriteLine("INPUT TO HIDDEN");
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    Console.WriteLine(i + " " + j + " " + inputToHiddenWeights[i, j]);
                }
            }
            Console.WriteLine();

            Console.WriteLine("HIDDEN TO OUTPUT");
            for (int i = 0; i < 2; i++)
            {
                Console.WriteLine(i + " " + hiddenToOutputWeights[i, 0]);
            }
            Console.WriteLine();

            Console.WriteLine("HIDDEN BIASES");
            for (int i = 0; i < 2; i++)
            {
                Console.WriteLine(i + " " + hiddenBiases[i]);
            }
            Console.WriteLine();

            Console.WriteLine("OUTPUT BIASES");
            Console.WriteLine(outputBias);
            
            Console.WriteLine();

            Console.WriteLine("SUCCESSES " + ((float)successes / 250f));

            return 0;
        }
    }
}
