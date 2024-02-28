using System;
using System.Collections.Generic;
using System.IO;

namespace DigitRecognition
{
    class Program
    {

        static void Main(string[] args)
        {
            /*
            {
                AdjustableNetwork network = new AdjustableNetwork(new int[] { 24, 24 }, 0.04f);

                network.Train();
                network.Test();
            }

            Console.WriteLine("FINISH");
            */
            
            float[,] succ = new float[6,6];

            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    Console.WriteLine(i + " " + j);
                    AdjustableNetwork network = new AdjustableNetwork(new int[] { 32,8 + i * 8, 8 + j * 8 }, 0.08f);

                    network.Train();
                    succ[i, j] = network.Test();
                }
            }


            using (StreamWriter writer = new StreamWriter("32nodesthreelayers.csv"))
            {
                for (int x = 0; x < 6; x++)
                {
                    for (int y = 0; y < 6; y++)
                    {
                        writer.Write(succ[x, y] + ", ");
                    }

                    writer.WriteLine();
                }
            }
            
            
            Console.WriteLine("FINISH");
            Console.ReadLine();
        }

    }

}
