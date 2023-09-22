import torch
import numpy as np
from torchvision import datasets, transforms
from tqdm import trange
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import sys
from torch.utils.mobile_optimizer import optimize_for_mobile

torch.cuda.device(0)

def tensor_rand_range(*shape, low = -1.0, high = 1.0) -> torch.Tensor:
    return torch.add(torch.mul(torch.rand(*shape), high - low), low)

def tensor_rand_scaled(*shape) -> torch.Tensor:
    return torch.mul(tensor_rand_range(*shape), np.power(np.prod(shape), -0.5))

class SqueezeExciteBlock2D(torch.nn.Module):
    def __init__(self, filters):
        super(SqueezeExciteBlock2D, self).__init__()
        self.filters = filters
        self.l1 = torch.nn.Linear(self.filters, self.filters // 32)
        self.l2 = torch.nn.Linear(self.filters // 32, self.filters)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        block: torch.Tensor = torch.nn.functional.avg_pool2d(input, (input.shape[2], input.shape[3]))
        block = torch.reshape(block, (-1, self.filters))
        
        block = self.l1(block)
        block = torch.relu(block)
        
        block = self.l2(block)
        block = torch.sigmoid(block)
        
        block = torch.reshape(block, (-1, self.filters, 1, 1))
        block = torch.mul(block, input)
        
        return block

class ConvBlock(torch.nn.Module):
    def __init__(self, h, w, inp, filters = 128, conv = 3):
        super(ConvBlock, self).__init__()
        self.h = h
        self.w = w
        self.inp = inp
        self.cweights = torch.nn.ParameterList([torch.randn(filters, inp if i == 0 else filters, conv, conv) for i in range(3)])
        self.cbiases = torch.nn.ParameterList([torch.randn(1, filters, 1, 1) for _ in range(3)])
        self._bn = torch.nn.BatchNorm2d(128)
        self._seb = SqueezeExciteBlock2D(filters)
        self._padder = torch.nn.ZeroPad2d(1)

    def forward(self, input) -> torch.Tensor:
        block = torch.reshape(input, (-1, self.inp, self.w, self.h))

        for cweight, cbias in zip(self.cweights, self.cbiases):
            block = self._padder(block)
                    
            block = torch.conv2d(block, cweight)
            block = torch.add(block, cbias)
            block = torch.relu(block)

        block = self._bn(block)
        block = self._seb(block)

        return block

class BigConvNet(torch.nn.Module):
    def __init__(self):
        super(BigConvNet, self).__init__()
        self.conv = torch.nn.ParameterList([ConvBlock(28, 28, 1), ConvBlock(28, 28, 128), ConvBlock(14, 14, 128)])
        self.l1 = torch.nn.Linear(128, 10, bias=False)
        self.l2 = torch.nn.Linear(128, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.cuda()
        x = self.conv[0](x)
        x = self.conv[1](x)

        x = torch.nn.functional.avg_pool2d(x, (2, 2))
        x = self.conv[2](x)

        x1 = torch.nn.functional.avg_pool2d(x, (14, 14))
        x1 = torch.reshape(x1, (-1, 128))

        x2 = torch.nn.functional.max_pool2d(x, (14, 14))
        x2 = torch.reshape(x2, (-1, 128))

        x1 = self.l1(x1)
        x2 = self.l2(x2)

        out = torch.add(x1, x2)
        return out

def train_one_epoch(epoch_index, writer, trainloader, model, optimizer, loss_fn):
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f"  batch {i + 1} loss: {last_loss}")
            tb_x = epoch_index * len(trainloader) + i + 1
            writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0

    return last_loss

def test_loaded_model(model_path: str):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = datasets.MNIST('mnist_train', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 32, shuffle = False)
    
    model = BigConvNet()
    model.cuda()
    model.load_state_dict(torch.load(model_path))

    loss_fn = torch.nn.CrossEntropyLoss()
    running_vloss = 0.0
    max_vloss = float()
    min_vloss = 1.0
    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(testloader):
            vinputs, vlabels = vdata
            vinputs = vinputs.cuda()
            vlabels = vlabels.cuda()

            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)

            running_vloss += vloss
            max_vloss = max(max_vloss, vloss)
            min_vloss = min(min_vloss, vloss)

    avg_vloss = running_vloss / (i + 1)
    print(f"Average vloss: {avg_vloss} | Max vloss: {max_vloss} | Min vloss: {min_vloss}")

def serialize_model(model_path: str):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = datasets.MNIST('mnist_train', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 32, shuffle = False)

    model = BigConvNet()
    model.cuda()
    model.load_state_dict(torch.load(model_path))

    model.eval()
    example_input, _ = next(iter(testloader))
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter("./out/model.ptl")

def main():
    writer = SummaryWriter()
    learning_rates = [1e-3, 1e-4, 1e-5, 1e-5]
    epochss = [13, 3, 3, 1]
    batch_size = 32

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
    testset = datasets.MNIST('mnist_train', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle = True)
    testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle = False)
    
    steps = len(trainset.train_data) // batch_size
    model = BigConvNet()
    model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0
    best_vloss = 1_000_000.

    for learning_rate, epochs in zip(learning_rates, epochss):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            print(f"EPOCH {epoch_number}")

            model.train(True)
            avg_loss = train_one_epoch(epoch_number, writer, trainloader, model, optimizer, loss_fn)

            running_vloss = 0.0
            model.eval()

            with torch.no_grad():
                for i, vdata in enumerate(testloader):
                    vinputs, vlabels = vdata
                    vinputs = vinputs.cuda()
                    vlabels = vlabels.cuda()

                    voutputs = model(vinputs)
                    vloss = loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print(f"LOSS train {avg_loss} valid {avg_vloss}")  

            writer.add_scalars("Training vs. Testing Loss",
                { "Training" : avg_loss, "Testing" : avg_vloss },
                epoch_number + 1)
            writer.flush()
            
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)

            epoch_number += 1

if __name__ == "__main__":
    serialize_model(sys.argv[1])