using System;
using System.Collections.Generic;
using System.Text;

namespace back_propagation
{
    internal class Node
    {
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
            decimal sum = 0;
            foreach (var node in nodes)
            {
                sum += node.x * node.weights[mynum];
            }
            x = sum;
        }

        public void CalculateReLU(Node[] nodes, int mynum)
        {
            decimal sum = 0;
            foreach (var node in nodes)
            {
                sum += (node.x > 0 ? node.x : 0) * node.weights[mynum];
            }
            x = sum;
        }
    }
}
