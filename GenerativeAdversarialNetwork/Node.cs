using System;
using System.Collections.Generic;
using System.Text;

namespace back_propagation
{
    internal class Node
    {
        private const double ActivationClamp = 1_000_000.0;

        public decimal[] weights { get; set; }
        public decimal x { get; set; }

        public Node(decimal[] weights)
        {
            this.weights = weights;
        }

        public Node()
        {
            this.weights = new decimal[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        }

        public void Calculate(Node[] nodes, int mynum)
        {
            double sum = 0.0;
            foreach (var node in nodes)
            {
                sum += Convert.ToDouble(node.x) * Convert.ToDouble(node.weights[mynum]);
            }

            sum = Math.Clamp(sum, -ActivationClamp, ActivationClamp);
            x = Convert.ToDecimal(sum);
        }

        public void CalculateReLU(Node[] nodes, int mynum)
        {
            double sum = 0.0;
            foreach (var node in nodes)
            {
                double relu = Math.Max(Convert.ToDouble(node.x), 0.0);
                sum += relu * Convert.ToDouble(node.weights[mynum]);
            }

            sum = Math.Clamp(sum, -ActivationClamp, ActivationClamp);
            x = Convert.ToDecimal(sum);
        }
    }
}
