from torch.utils.data import DataLoader
from cifar import load_cifar, SampleCNN, SimpleTrainer, load_mnist
from torch2cmsis import converter
from quantize_utils import CMSISConverter, inference

CONFIG = {
    "batch_size": 64,
    "epochs": 5,
    "learning_rate": 0.001,
    "learning_step": 5000,
    "learning_gamma": 0.99,
    "name": "sample_model",
    "shape": (1, 28, 28),
    "dataset": load_mnist
}   


def train_cifar(config):
    datasets = config["dataset"]()
    dataloaders =  {
            i: DataLoader(
                sett,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=4
                )
            for i, sett in zip(["train", "val", "test"], datasets)
        }
    cnn = SampleCNN(shape=config["shape"], batch_size = config["batch_size"])
    trainer = SimpleTrainer(
        datasets=datasets,
        dataloaders=dataloaders
    )
    cnn = trainer.train(cnn, config, config.get("name"))
    accuracy_test = trainer.evaluate(cnn)
    cnn.to('cpu')
    cnn.eval()
    
    print(accuracy_test)
    cm_converter = CMSISConverter("cfiles", cnn, "weights.h", "parameters.h")
    input, label, pred = inference(cnn, dataloaders["train"])
    input.to('cpu')
    print(label, pred)
    cm_converter.quantize_input(input[0])
    cm_converter.generate_intermediate_values(input)
    cm_converter.convert_model_cmsis()
    cm_converter.evaluate_cmsis("main", dataloaders['test'])

if __name__ == "__main__":
    train_cifar(CONFIG)