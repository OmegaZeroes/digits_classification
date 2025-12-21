import torch
import seaborn as sns 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
from src.data.dataloader import getDataSet, DataCollator, getDataTest
from omegaconf import  OmegaConf
import torch.nn as nn
import torch.optim as optim
from save_samples import testImage
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from src.models.model import MLP, DigitsClassifier, EarlyStopping
from torchvision.datasets import MNIST

config = OmegaConf.load("./configs/config.yaml")
collator = DataCollator()
train_dataset, val_dataset, test_dataset,total_size = getDataSet(config.data.root_dir)

loaders = {
    'train': DataLoader(
    dataset=train_dataset,
    batch_size=config.data.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collator),

    'val': DataLoader(
    dataset=val_dataset,
    batch_size=config.data.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collator),

    'test': DataLoader(
    dataset=test_dataset,
    batch_size=config.data.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collator),

}

    

#TODO: Define device to train on, loss function, optimizer.
#TODO: Initialize the model, and load configs(learning rate and epochs) 

early_stopping = EarlyStopping(patience=5, verbose=True)

epochs = config.trainer.epochs

learning_rate = config.trainer.learning_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

metric_params = {'num_classes': 10, 'average': 'macro'}

precision_metric = MulticlassPrecision(**metric_params).to(device)
recall_metric = MulticlassRecall(**metric_params).to(device)
f1_metric = MulticlassF1Score(**metric_params).to(device)


loss_fn = nn.CrossEntropyLoss()

writer = SummaryWriter(log_dir="runs/my_experiment") #Tensor Board


def checkDataLoader() -> None:
    print("Load data completed!!!")
    print(f"Total image: {total_size}")
    print(f"Training: {len(train_dataset)}")        
    print(f"Validation: {len(val_dataset)}")    
    print(f"Testing: {len(test_dataset)}")

    print("<---Training Starting--->")
    print (f"Using device: {device}")
    

def train(model, optimizer) -> None:     
    print("<----Start Training----->")
    
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()

        for i,(images, labels) in enumerate(loaders['train']):
            images = images.to(device)
            labels = labels.to(device)
            target_indices = torch.argmax(labels, dim=1)

            outputs = model(images)
            loss = loss_fn(outputs, target_indices)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i+1}/{len(loaders['train'])}], Loss: {loss.item():.4f}")
        avg_train_loss = running_loss / len(loaders['train'])
        #Starting validation progress
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()

        total_val_steps = len(loaders['val'])
        print(f"--- Starting Validation for Epoch [{epoch+1}/{epochs}] ---")
        with torch.no_grad():
            for i, (images, labels) in enumerate(loaders['val']):
                images = images.to(device)
                labels = labels.to(device)
            
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                _, true_labels = torch.max(labels.data, 1)

                precision_metric.update(predicted, true_labels)
                recall_metric.update(predicted, true_labels)
                f1_metric.update(predicted, true_labels)

                total += labels.size(0)
                correct += (predicted == true_labels).sum().item()

           
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Val Step [{i+1}/{total_val_steps}], Val Loss: {loss.item():.4f}")

        final_precision = precision_metric.compute()
        final_recall = recall_metric.compute()
        final_f1 = f1_metric.compute()


        avg_val_loss = val_loss / total_val_steps

        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        writer.add_scalars('Loss', {
        'train': avg_train_loss,
        'val': avg_val_loss
        }, epoch)

        print(f"End of Epoch {epoch+1} -> Avg Val Loss: {avg_val_loss:.4f}, F1: {final_f1.item():.4f}, Precision: {final_precision.item():.4f}, Recall: {final_recall.item():.4f}")
        print("--------------------------------------------------")
    writer.close()

def test(model)-> None:
    print("<---Begin Test Process--->")
    model.load_state_dict(torch.load("model.pth"))
    test_loss = 0.0
    correct = 0
    total = 0
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    model.eval()
    with torch.no_grad():
        for images, labels in loaders['test']:
            images, labels = images.to(device), labels.to(device)

            true_labels = torch.max(labels.data, 1)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(labels.data, 1)

            precision_metric.update(predicted, true_labels)
            recall_metric.update(predicted, true_labels)
            f1_metric.update(predicted, true_labels)

            total += labels.size(0)
            correct += (predicted==true_labels).sum().item()
    avg_test_loss = test_loss / len(loaders['test'])
    test_acc = 100 * correct/total

    final_precision = precision_metric.compute()
    final_recall = recall_metric.compute()
    final_f1 = f1_metric.compute()
    

    print(f"Test Loss: {avg_test_loss}, Accuaracy: {test_acc}, F1: {final_f1.item():.4f}, Precision: {final_precision.item():.4f}, Recall: {final_recall.item():.4f}")

def plot_confusion_matrix(model) -> None:
    print("<--- Plot Confusion Matrix (TEST SET) --->")
    with open('model.pth', 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for images, labels in loaders['test']:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(labels, dim=1)

            all_preds.append(preds)
            all_true.append(true_labels)

    all_preds = torch.cat(all_preds)
    all_true = torch.cat(all_true)
    # Confusion Matrix (10 classes)
    cm = torch.zeros(10, 10, dtype=torch.int64)

    for t, p in zip(all_true, all_preds):
        cm[t, p] += 1

    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm.numpy(),
        annot=True,
        fmt='d',
        cmap='Blues'
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix - Digit Classification")
    plt.show()


test_data = getDataTest(root_dir=config.data.root_dir)

test_dataset = MNIST(root=config.data.root_dir, train=False)
def testImage() -> list:
    result = []
    for i, (image, label) in enumerate(test_dataset):
        if i < 5:
            image.save('{:1d}.jpg'.format(i))
            result.append(label)
        else:
            break
    return result


def run_test(model) -> None:
    label = testImage()
    with open('model.pth', 'rb') as f: 
        model.load_state_dict(torch.load(f))  
    plt.figure(figsize=(15, 3))
    for i in range(5):
        img = Image.open(f'{i}.jpg')

        plt.subplot(1, 5, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {label[i]}")
        plt.axis('off')

        img = Image.open(f'{i}.jpg')
        img_transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = img_transform(img).unsqueeze(0).to(device)
        output = model(img_tensor)
        pre_label = torch.argmax(output)
        print(f"Predicted label: {pre_label}")
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    checkDataLoader()
    success = False
    model = DigitsClassifier().to(device) #Default
    while success != True:
        choice = input("Type 1 to use MLP or 2 for CNN: ")
        if int(choice)==1:
            model = MLP().to(device)
            success = True
        elif int(choice)==2:
            success =True
        else:
            print("Typo!")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train(model, optimizer)
    test(model)
    plot_confusion_matrix(model)
    run_test(model)
    try:
        image, labels = next(iter(loaders['train']))
        print("\n---TEST DATLOADER---") #Data Loader test
        print(f"Size image batch: {image.shape}")
        print(f"Size labels batch: {labels.shape}")
        print(f"labels in batch: {labels[:5]}")
    except Exception as e:
        print(f"Error when get batch: {e}")