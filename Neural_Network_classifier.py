import pandas as pd 
from tqdm import tqdm
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class HARDataset(Dataset):
    def __init__(self,ds):
        self.df=pd.read_csv(ds) # "train.csv" / "test.csv"
        map_function={'STANDING':0, 'SITTING':1, 'LAYING':2, 'WALKING':3, 'WALKING_DOWNSTAIRS':4,'WALKING_UPSTAIRS':5}
        self.df['Activity']=self.df['Activity'].map(map_function)
        x=self.df.iloc[:,:562]
        y=self.df.iloc[:,562]
        self.x=torch.tensor(x.values,dtype=torch.float)
        self.y=torch.tensor(y.values,dtype=torch.long)
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]


class NNClassifier(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.l1=nn.Linear(input_size,4096)
        self.l2=nn.Linear(4096,2048)
        self.l3=nn.Linear(2048,512)
        self.l4=nn.Linear(512,64)
        self.l5=nn.Linear(64,output_size)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.relu(self.l1(x))
        x=self.relu(self.l2(x))
        x=self.relu(self.l3(x))
        x=self.relu(self.l4(x))
        return self.l5(x)



train_dataset=HARDataset("train.csv")
test_dataset=HARDataset("test.csv")
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=True)

device='cuda:4' if torch.cuda.is_available() else 'cpu'
print(f"using device {device}")
n_features=562
n_classes=6
n_epochs=10
model=NNClassifier(n_features,n_classes).to(device)
criterian=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)

for epoch in range(n_epochs):
    loop= tqdm(train_loader,total=len(train_loader),desc=f'Epoch {epoch + 1}/{n_epochs}',leave=False)
    for x,y in loop:
        x=x.to(device)
        y=y.to(device)
        y_pred=model(x)
        loss=criterian(y_pred,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def accuracy1(model,test_laoder,device):
    model.eval()
    count=0
    correct=0

    with torch.no_grad():
        for x,y in test_laoder:
            x=x.to(device)
            y=y.to(device)
            y_pred=model(x)
            _,predicted=torch.max(y_pred,dim=-1)
            count += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy=correct/count
    return accuracy
test_accuracy = accuracy1(model, test_loader, device)
print(f'Test Accuracy: {test_accuracy:.2f}%')

