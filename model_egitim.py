import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import copy
from PIL import ImageFile
from datetime import datetime
from torch.multiprocessing import freeze_support
import sys  

ImageFile.LOAD_TRUNCATED_IMAGES = True

def progress_bar(current, total, bar_length=50, phase='', loss=None, acc=None):
    percent = float(current) / total
    arrow = '=' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(arrow))
    
    status = ""
    if loss is not None and acc is not None:
        status = f"Loss: {loss:.4f}, Acc: {acc:.4f}"
    
    sys.stdout.write(f"\r[{arrow + spaces}] {int(round(percent * 100))}% {phase} {current}/{total} {status}")
    sys.stdout.flush()
    
    if current == total:
        sys.stdout.write('\n')

def train_model(model, criterion, optimizer, scheduler, num_epochs=30, patience=7):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    early_stopping_counter = 0
    best_val_loss = float('inf')
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,learning_rate\n")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  
                print(f"Eğitim aşaması başladı...")
            else:
                model.eval()   
                print(f"Değerlendirme aşaması başladı...")
            
            running_loss = 0.0
            running_corrects = 0
            batch_count = 0
            
            total_batches = len(dataloaders[phase])
            
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                batch_count += 1
                batch_start_time = time.time()
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                batch_loss = loss.item()
                batch_corrects = torch.sum(preds == labels.data).item()
                batch_acc = batch_corrects / inputs.size(0)
                
                running_loss += batch_loss * inputs.size(0)
                running_corrects += batch_corrects
                
                current_loss = running_loss / (batch_count * inputs.size(0))
                current_acc = running_corrects / (batch_count * inputs.size(0))
                
                batch_time = time.time() - batch_start_time
                
                progress_bar(batch_idx + 1, total_batches, phase=phase, loss=current_loss, acc=current_acc)
                
                print(f"\nBatch {batch_idx+1}/{total_batches} | "
                      f"Batch Süresi: {batch_time:.2f}s | "
                      f"Batch Loss: {batch_loss:.4f} | "
                      f"Batch Acc: {batch_acc:.4f} | "
                      f"Anlık Loss: {current_loss:.4f} | "
                      f"Anlık Acc: {current_acc:.4f}")
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            print(f'\n{phase} Epoch Özeti - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)
                
                scheduler.step(epoch_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Öğrenme oranı: {current_lr:.8f}")
                
                with open(log_file, 'a') as f:
                    f.write(f"{epoch+1},{train_losses[-1]:.6f},{train_accs[-1]:.6f},{val_losses[-1]:.6f},{val_accs[-1]:.6f},{current_lr:.8f}\n")
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model_path = os.path.join(model_dir, f'best_model_{timestamp}.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'acc': epoch_acc,
                    }, best_model_path)
                    print(f"Yeni en iyi model kaydedildi: {best_model_path}")
                
                epoch_model_path = os.path.join(model_dir, f'model_epoch_{epoch+1}_{timestamp}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'acc': epoch_acc,
                }, epoch_model_path)
                print(f"Epoch modeli kaydedildi: {epoch_model_path}")
                
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    print(f'EarlyStopping sayacı: {early_stopping_counter}/{patience}')
                    if early_stopping_counter >= patience:
                        print('Early stopping! Eğitim durduruldu.')
                        time_elapsed = time.time() - since
                        print(f'Eğitim tamamlandı - Süre: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                        print(f'En iyi doğrulama başarımı: {best_acc:.4f}')
                        
                        model.load_state_dict(best_model_wts)
                        return model, train_losses, train_accs, val_losses, val_accs
        
        elapsed_time = time.time() - since
        estimated_time = elapsed_time / (epoch + 1) * (num_epochs - epoch - 1)
        print(f"\nEpoch {epoch+1}/{num_epochs} tamamlandı - "
              f"Geçen süre: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s - "
              f"Tahmini kalan süre: {estimated_time // 60:.0f}m {estimated_time % 60:.0f}s")
        print('-' * 50)
    
    time_elapsed = time.time() - since
    print(f'Eğitim tamamlandı - Toplam süre: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'En iyi doğrulama başarımı: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    return model, train_losses, train_accs, val_losses, val_accs

def evaluate_model(model, dataloader):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Test değerlendirmesi başladı...")
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar(batch_idx + 1, total_batches, phase="Test")
    
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return accuracy, conf_matrix

if __name__ == '__main__':
    freeze_support() 
    
    start_time = time.time()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    train_dir = "./dataset/train"
    test_dir = "./dataset/test"
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
        
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(train_dir), data_transforms['train']),
        'test': datasets.ImageFolder(os.path.join(test_dir), data_transforms['test'])
    }
        
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=12, pin_memory=True),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=12, pin_memory=True)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    print(f"Sınıflar: {class_names}")
    print(f"Eğitim seti boyutu: {dataset_sizes['train']} görüntü")
    print(f"Test seti boyutu: {dataset_sizes['test']} görüntü")
    print(f"Eğitim batch sayısı: {len(dataloaders['train'])}")
    print(f"Test batch sayısı: {len(dataloaders['test'])}")
        
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.csv")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\nEğitim başlıyor... (Timestamp: {timestamp})")
    print(f"Log dosyası: {log_file}")
    print('-' * 50)
    
    model, train_losses, train_accs, val_losses, val_accs = train_model(
        model, criterion, optimizer, scheduler, num_epochs=30, patience=7)
    
    print("\nEğitim grafikleri oluşturuluyor...")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    curves_path = os.path.join(log_dir, f'training_curves_{timestamp}.png')
    plt.savefig(curves_path)
    print(f"Eğitim grafikleri kaydedildi: {curves_path}")
    plt.show()
    
    print("\nTest değerlendirmesi yapılıyor...")
    
    test_accuracy, test_conf_matrix = evaluate_model(model, dataloaders['test'])
    print(f'\nTest Accuracy: {test_accuracy:.4f}')
    print('Confusion Matrix:')
    print(test_conf_matrix)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(test_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = test_conf_matrix.max() / 2.
    for i in range(test_conf_matrix.shape[0]):
        for j in range(test_conf_matrix.shape[1]):
            plt.text(j, i, format(test_conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if test_conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    conf_matrix_path = os.path.join(log_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(conf_matrix_path)
    print(f"Karışıklık matrisi kaydedildi: {conf_matrix_path}")
    plt.show()
    
    summary = {
        'timestamp': timestamp,
        'best_train_acc': max(train_accs),
        'best_val_acc': max(val_accs),
        'final_test_acc': test_accuracy,
        'model_path': os.path.join(model_dir, f'best_model_{timestamp}.pth'),
        'log_path': log_file
    }
    
    summary_file = os.path.join(log_dir, 'training_summary.csv')
    summary_exists = os.path.exists(summary_file)
    
    with open(summary_file, 'a') as f:
        if not summary_exists:
            f.write("timestamp,best_train_acc,best_val_acc,final_test_acc,model_path,log_path\n")
        f.write(f"{summary['timestamp']},{summary['best_train_acc']:.6f},{summary['best_val_acc']:.6f},{summary['final_test_acc']:.6f},{summary['model_path']},{summary['log_path']}\n")
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\nÖzet:")
    print(f"Toplam süre: {int(hours)}s {int(minutes)}d {int(seconds)}s")
    print(f"En iyi eğitim başarımı: {summary['best_train_acc']:.4f}")
    print(f"En iyi doğrulama başarımı: {summary['best_val_acc']:.4f}")
    print(f"Test başarımı: {summary['final_test_acc']:.4f}")
    print(f"Özet dosyası: {summary_file}")
    
    print("\nEğitim tamamlandı. Tüm sonuçlar kaydedildi.")
