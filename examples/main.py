from torch.data.utils import DataLoader
import cifar

CONFIG = {
    "batch_size": 8,
    "epochs": 25,
    "learning_rate": 0.001,
    "learning_step": 5000,
    "learning_gamma": 0.99,
    "name": "sample_model.pth"
}


def train_cifar(config):
    datasets = cifar.load_cifar()
    dataloaders =  {
            i: DataLoader(
                sett,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=4
                )
            for i, sett in zip(["train", "val", "test"], datasets)
        }
    cnn = cifar.SampleCNN(config["batch_size"])
    trainer = cifar.SimpleTrainer(
        datasets=datasets,
        dataloaders=dataloaders
    )
    cnn = trainer.train(cnn, config, config.get("name"))
    accuracy_test = trainer.evaluate(cnn)
    print(accuracy_test)

    