# 필요한 패키지 임포트 
import os 
import torch
import torch.nn as nn 
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam

# 하이퍼파라메터 선언 
lr = 0.001
image_size = 28
num_classes = 10 
hidden_size = 500 
batch_size = 100 
epochs = 3 
results_folder = 'results'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 저장 
# 상위 저장 폴더를 만들어야 함 
if not os.path.exists(results_folder): 
    os.makedirs(results_folder)
# 내가 저장을 할 하위 폴더를 만들어야 함 (하위 폴더가 앞으로 사용될 타겟 폴더가 됨) 
    target_folder_name = max([0] + [int(e) for e in os.listdir(results_folder)])+1
    target_folder = os.path.join(results_folder, str(target_folder_name))
    os.makedirs(target_folder)
# 타겟 폴더 밑에 hparam 저장 (text의 형태로)

# 모델 설계도 그리기 
class MLP(nn.Module):
    def __init__(self, image_size, hidden_size, num_classes) : # 레고 조각 생성 
        super().__init__()
        self.image_size = image_size
        self.mlp1 = nn.Linear(image_size * image_size, hidden_size) 
        self.mlp2 = nn.Linear(hidden_size, hidden_size) 
        self.mlp3 = nn.Linear(hidden_size, hidden_size) 
        self.mlp4 = nn.Linear(hidden_size, num_classes) 

    def forward(self, x): # x : [batch_size, 28, 28, 1]  # 레고 조각을 조립 
        batch_size = x.shape[0]
        x = torch.reshape(x, (-1, self.image_size * self.image_size)) # [batch_size, 28*28]
        x = self.mlp1(x) # [batch_size, 500]
        x = self.mlp2(x) # [batch_size, 500]
        x = self.mlp3(x) # [batch_size, 500]
        x = self.mlp4(x) # [batch_size, 10]
        return x 

# 설계도를 바탕으로 모델을 만들어야 함 <- 하이퍼파라메터 사용 
myMLP = MLP(image_size, hidden_size, num_classes).to(device)

# 데이터 불러오기 
# dataset 설정 
train_mnist = MNIST(root='../../data/mnist', train=True, transform=ToTensor(), download=True)
test_mnist = MNIST(root='../../data/mnist', train=False, transform=ToTensor(), download=True)
# dataloader 설정 
train_loader = DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=False)

# Loss 선언 
loss_fn = nn.CrossEntropyLoss()
# Optimizer 선언 
optim = Adam(params=myMLP.parameters(), lr=lr) 

# 평가 함수 구현 
def evaluate(model, loader, device):
    with torch.no_grad(): 
        model.eval()
        total = 0 
        correct = 0 
        for images, targets in loader: 
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            output_index = torch.argmax(output, dim=1) 
            total += targets.shape[0]
            correct += (output_index == targets).sum().item()

    acc = correct / total * 100 
    model.train()
    return acc 

def evaluate_by_class(model, loader, device, num_classes): 
    with torch.no_grad(): 
        model.eval()
        correct = torch.zeros(num_classes)
        total =  torch.zeros(num_classes)
        for images, targets in loader: 
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            output_index = torch.argmax(output, dim=1) 
            for _class in range(num_classes): 
                total[_class] += (targets == _class).sum().item()
                correct[_class] += ((targets == _class) * (output_index == _class)).sum().item()

    acc = correct / total * 100  # shape : [10]
    model.train()
    return acc 

# 학습을 위한 반복 (Loop) for / while 
for epoch in range(epochs): 
    # 입력할 데이터를 위해 데이터 준비 (dataloader) 
    for idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # 모델에 데이터를 넣기 
        output = myMLP(images)
        # 모델의 출력과 정답을 비교하기 (Loss 사용) 
        loss = loss_fn(output, targets)
        # Loss를 바탕으로 업데이트 진행 (Optimizer) 
        loss.backward()
        optim.step()
        optim.zero_grad()

        if idx % 100 == 0: 
            print(loss)
            # 평가(로깅, print), 저장 
            acc = evaluate(myMLP, test_loader, device)
            # acc = evaluate_by_class(myMLP, test_loader, device, num_classes)
            # 평가 결과가 좋으면 타겟 폴더에 모델 weight 저장을 진행 
            # 평가 결과가 좋다는게 무슨 의미지? -> 과거의 평가 결과보다 좋은 수치가 나오면 결과가 좋다고 얘기합니다. 
            # 과거 결과(max) < 지금 결과(acc)
            if _max < acc :
                print('새로운 acc 등장, 모델 weight 업데이트',acc)
                _max = acc
                torch.save(
                    myMLP.state_dict(),
                    os.path.join(target_folder,'myMLP_best.ckpt')
                )