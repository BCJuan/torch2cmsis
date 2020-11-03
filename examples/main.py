from torch.utils.data import DataLoader
from cifar import load_cifar, SampleCNN, SimpleTrainer
from torch2cmsis import converter


CONFIG = {
    "batch_size": 8,
    "epochs": 1,
    "learning_rate": 0.001,
    "learning_step": 5000,
    "learning_gamma": 0.99,
    "name": "sample_model"
}


def train_cifar(config):
    datasets = load_cifar()
    dataloaders =  {
            i: DataLoader(
                sett,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=4
                )
            for i, sett in zip(["train", "val", "test"], datasets)
        }
    cnn = SampleCNN(batch_size = config["batch_size"])
    trainer = SimpleTrainer(
        datasets=datasets,
        dataloaders=dataloaders
    )
    cnn = trainer.train(cnn, config, config.get("name"))
    accuracy_test = trainer.evaluate(cnn)
    print(accuracy_test)
    cmsis_converter = converter.CMSISConverter(
        cnn,
        "weights.h",
        "parameters.h",
        "input.h",
        "logging.h",
        weight_bits=8
    )
    input, label, pred = converter.inference(cnn, dataloaders['val'])
    input.to('cpu')
    cnn.to('cpu')
    cnn.eval()
    cmsis_converter.quantize_input(input[0])
    cmsis_converter.generate_intermediate_values(input, cnn)
    cmsis_converter.convert_model_cmsis(cnn)

if __name__ == "__main__":
    train_cifar(CONFIG)