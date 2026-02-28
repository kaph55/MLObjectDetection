using System;
using System.IO;
using Microsoft.ML;

namespace ObjectConsole.Wpf
{
    public class ModelLoader
    {
        private readonly MLContext _mlContext;

        public ITransformer? Model { get; private set; }
        public DataViewSchema Schema { get; private set; } = null!;

        public ModelLoader(MLContext mlContext)
        {
            _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        }

        public void Load(string modelPath)
        {
            if (!File.Exists(modelPath)) throw new FileNotFoundException("Model not found", modelPath);
            using var fs = File.OpenRead(modelPath);
            Model = _mlContext.Model.Load(fs, out var schema);
            Schema = schema;
        }

        public IDataView Transform(IDataView input)
        {
            if (Model == null) throw new InvalidOperationException("Model not loaded.");
            return Model.Transform(input);
        }
    }
}