from omegaconf import OmegaConf
import numpy as np

import torch
import torch.nn as nn
import torchvision

from models import naive_dec
from utils import parse_args, load_yaml_config
import our_dataset as our_dataset

if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(config_filename=args.config)
    config = OmegaConf.create(config)

    root = args.root if args.root is not None else "outputs/regularizeITMORE_7sbjs_drop0.5_20250424"
    root = root + "/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = []
    subjects = [2, 4, 5, 6, 7, 10, 11]
    for i in subjects:
        datasets.append(our_dataset.meg_dataset(config=config, s=i, train=False))

    dataset = torch.utils.data.ConcatDataset(datasets)

    print("Expected Number of samples:", 400 * 1 * len(subjects), "Actual Number of samples:", len(dataset))

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    backbone = torchvision.models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = naive_dec.NaiveModel(backbone, num_subjects=len(subjects), config=config).to(device)
    model.load_state_dict(torch.load(root + 'best_model.pth', map_location=device))
    model = model.to(device)

    model.eval()
    all_test_outputs, all_test_labels, all_subjects = [], [], []
    with torch.no_grad():
        subject_accuracies = {}
        for data, target, sbjs in test_dataloader:
            target = [0 if i == 43 else 1 for i in target]
            data, target = data.to(device), torch.tensor(target).to(device)
            
            output = model(data, sbjs, subjects)
            _, predicted = torch.max(output.data, 1)

            all_test_outputs.append(output.cpu())
            all_test_labels.append(target.cpu())
            all_subjects.append(sbjs.cpu())
            
            for i, subj in enumerate(sbjs):
                if subj.item() not in subject_accuracies:
                    subject_accuracies[subj.item()] = {'correct': 0, 'total': 0}
                
                subject_accuracies[subj.item()]['total'] += 1
                if predicted[i] == target[i]:
                    subject_accuracies[subj.item()]['correct'] += 1
        
    for subj, stats in subject_accuracies.items():
        accuracy = 100 * stats['correct'] / stats['total']
        print(f'Accuracy for Subject {subj}: {accuracy:.2f}%')

    total_correct = sum(stats['correct'] for stats in subject_accuracies.values())
    total = sum(stats['total'] for stats in subject_accuracies.values())
    overall_accuracy = 100 * total_correct / total
    print(f'Overall Accuracy on Test Set: {overall_accuracy:.2f}%')
    
    torch.save({"outputs": torch.cat(all_test_outputs, dim=0), 
                "labels": torch.cat(all_test_labels, dim=0),
                "subjects": torch.cat(all_subjects, dim=0)},
               root + 'test_outputs.pt')


