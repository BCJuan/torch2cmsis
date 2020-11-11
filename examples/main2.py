from torch.utils.data import DataLoader
from cifar import load_cifar, SampleCNN, SimpleTrainer, load_mnist
from torch2cmsis import converter
from quantize_utils import CMSISConverter

CONFIG = {
    "batch_size": 64,
    "epochs": 5,
    "learning_rate": 0.001,
    "learning_step": 5000,
    "learning_gamma": 0.99,
    "name": "sample_model",
    "shape": (1, 28, 28),
    "dataset": load_mnist,
    "compilation": 'gcc -g -I../../CMSIS_5/CMSIS/Core/Include \
            -I../../CMSIS_5/CMSIS/DSP/Include \
            -I../../CMSIS_5/CMSIS/NN/Include \
            -D__ARM_ARCH_8M_BASE__ \
            ../../CMSIS_5/CMSIS/NN/Source/*/*.c \
            ../../CMSIS_5/CMSIS/DSP/Source/StatisticsFunctions/arm_max_q7.c \
            main.c -o main',
    "exec_path": "main"
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
    cm_converter = CMSISConverter("cfiles", cnn, "weights.h", "parameters.h",
                                  8, config.get("compilation"))
    cm_converter.generate_intermediate_values(dataloaders['val'])
    cm_converter.convert_model_cmsis()
    cm_converter.evaluate_cmsis(config.get("exec_path"), dataloaders['test'])
    input, label = next(iter(dataloaders['test']))
    cm_converter.sample_inference_checker(config.get("exec_path"), input)

if __name__ == "__main__":
    train_cifar(CONFIG)