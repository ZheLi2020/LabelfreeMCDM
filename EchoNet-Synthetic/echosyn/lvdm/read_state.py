import os
import numpy as np
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    epochs = list(range(1, 1000001, 10000))
    # losses = []

    # Authenticate
    api = wandb.Api()

    # Get your run (replace with your entity/project/run_id)
    run = api.run("lizhesz/motion/kzpval3t")

    # Download history (all logged values)
    history = run.history(keys=["train_loss"])  # specify "loss" or other metrics

    # Load loss
    # loss_values = history["train_loss"].tolist()
    # loss_values = np.array(loss_values)
    # min_loss = np.min(loss_values)
    # min_epoch = np.argmin(loss_values)
    loss_values = []
    for row in tqdm(run.scan_history(keys=["train_loss"])):
        if "train_loss" in row:
            loss_values.append(row["train_loss"])

    print(f"Extracted {len(loss_values)} values")

    loss_values = np.array(loss_values)
    losses = loss_values[epochs]
    min_loss = np.min(losses)
    min_epoch = np.argmin(losses)
    print(f'The min loss is {min_loss}, min index is {min_epoch}, min epoch is {epochs[min_epoch]}.')

    # draw figures
    plt.xticks(epochs)
    plt.title('Losses')
    plt.plot(epochs, losses)
    plt.savefig('losses.png')
    print('Done')


if __name__ == "__main__":

    main()
    # root_path = '/vol/idea_longterm/ot70igyn/EchoNet-Synthetic/experiments_un/20250804-234125/checkpoint-10000'
    #
    # data_path = os.path.join(root_path, 'random_states_0.pkl')
    #
    # with open(data_path, "rb") as files:
    #     random = pickle.load(files)
    #
    # print('Done')
