import logging
import sys
import time
import re

import click
import click_log
import tqdm

import multiprocessing as mp

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim


logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("mnist1")
click_log.basic_config(logger)


@click.command(name=logger.name)
@click_log.simple_verbosity_option(logger)
@click.option("-o", "--output-dir", required=True, type=click.Path(), help="Output path for data")
@click.option("-c", "--checkpoint", required=False, type=click.Path(), help="Resume training from previous model")
@click.option("-n", "--num-epochs", default=15, type=int, help="Resume training from previous model")
def main(output_dir, checkpoint, num_epochs):
    """MNIST hand-written digit recognition (model 1)."""

    t_start = time.time()
    logger.info("Invoked via: pytorch_examples %s", " ".join(sys.argv[1:]))

    train_set, test_set = fetch_data(output_dir)
    train_loader, test_loader = load_data(train_set, test_set)

    # display_images(train_loader, 60)

    model, first_epoch = build_model(checkpoint)
    last_epoch = train_model(train_loader, model, epochs=num_epochs, first_epoch=first_epoch)
    apply_model(test_loader, model)
    save_model(output_dir, model, last_epoch)

    t_end = time.time()
    logger.info(f"Done. Total time elapsed: {t_end - t_start:2.2f}s.")


def save_model(output_dir, model, last_epoch):
    model_out = f'{output_dir}/mnist_model.epoch{last_epoch}.pt'
    logger.info(f"Saving model to {model_out}")

    torch.save(model.state_dict(), model_out)


def build_model(checkpoint=None, input_size=784, hidden_sizes=[128, 64], output_size=10):
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LogSoftmax(dim=1)
    )

    first_epoch = 0
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
        model.eval()

        reg = r'epoch(\d+)'
        m = re.search(reg, checkpoint)
        first_epoch = int(m.group(1))

    logger.info(model)
    if first_epoch > 0:
        logger.info(f'Resuming from epoch {first_epoch}')
    
    return model, first_epoch


def apply_model(test_loader, model):
    correct_count, all_count = 0, 0
    for images,labels in test_loader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]

            if true_label == pred_label:
                correct_count += 1

            all_count += 1

    logger.info(f'Number of images tested = {all_count}')
    logger.info(f'Model accuracy = {correct_count/all_count}')


def train_model(train_loader, model, epochs=15, first_epoch=0):
    criterion = nn.NLLLoss()
    images, labels = next(iter(train_loader))
    images = images.view(images.shape[0], -1)

    logps = model(images)
    loss = criterion(logps, labels)

    time_training_start = time.time()

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    for e in range(first_epoch, first_epoch + epochs):
        running_loss = 0
        for images, labels in train_loader:
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        else:
            print(f'Epoch {e} - Training loss: {running_loss/len(train_loader)}')

    time_training_end = time.time()

    logger.info(f"Training time: {time_training_end - time_training_start:2.2f}s")

    return e


def display_images(train_loader, num_of_images = 60):
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    figure = plt.figure()
    for index in range(1, num_of_images + 1):
        plt.subplot(num_of_images/10, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

    plt.show()


def load_data(train_set, test_set):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    return train_loader,test_loader


def fetch_data(output_dir):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

    train_set = datasets.MNIST(output_dir, train=True, transform=transform, download=True)
    test_set = datasets.MNIST(output_dir, train=False, transform=transform, download=True)
    return train_set,test_set