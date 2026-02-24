using System;
using System.Collections.Generic;
using System.Text;

namespace back_propagation
{
    internal class ImageData
    {
        public string ImagePath { get; set; }
        public int Label { get; set; }
        public int FileName { get; set; }

        public ImageData(string imagePath, int label, int fileName)
        {
            ImagePath = imagePath;
            Label = label;
            FileName = fileName;
        }
    }
}
