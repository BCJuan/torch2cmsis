from torch.utils.data import DataLoader
from cifar import load_cifar, SampleCNN, SimpleTrainer, load_mnist
from torch2cmsis import converter


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
    cmsis_converter = converter.CMSISConverter(
        cnn,
        "cfiles",
        "weights.h",
        "parameters.h",
        8
    )
    cmsis_converter.prepare_quantization(dataloaders['train'])
    cmsis_converter.convert_model_cmsis()
    input, label, pred = converter.inference(cnn, dataloaders['val'])
    input.to('cpu')
    print(label, pred)
    cmsis_converter.register_logging(input)
    print("dasdasdad")
    # cmsis_converter.evaluate_cmsis("./main", dataloaders["test"])

if __name__ == "__main__":
    train_cifar(CONFIG)