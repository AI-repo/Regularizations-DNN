from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sampler import ImbalancedDatasetSampler
from resnet import resnet34
from utils import get_mean_and_std
#import models.imagenet as customized_models
from progress_dir.progress.bar import Bar 
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from torch.nn.functional import normalize
from sklearn.metrics import classification_report,precision_recall_fscore_support,f1_score
from torchsummary import summary

"""default_model_names = sorted(name for name in models.__dict__
			if name.islower() and not name.startswith("__")
			and callable(models.__dict__[name]))

print (default_model_names)"""

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='/home/amin/models/physionet/data/', type=str)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 1)')

# Optimization options
#parser.add_argument('--sencoe', default =33, type=float,  help=' The sensitivity regularization coefficient')
parser.add_argument('--lambdaa', default =0.7, type =float, metavar='lambda', help='Signifance value of the sensitivity regularization term')
parser.add_argument('--epochs', default=70, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=1, type=int, metavar='N', help='train batchsize (default: 10)')
parser.add_argument('--test-batch', default=1, type=int, metavar='N',  help='test batchsize (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--schedule', type=int, nargs='+', default=[40, 90,150], help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,	metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--ortho-decay', '--od', default=1e-3, type=float, help = 'ortho weight decay')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='/home/amin/models/resnet/', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: checkpoint)')
#/home/amin/models/resnet/model_best.pth.tar
parser.add_argument('--depth', type=int, default=28, help='Model depth.')
parser.add_argument('--widen-factor', type=int, default=2, help='Widen factor. 4 -> 64, 8 -> 128, ...')

# Miscs
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained',  dest='imagenet', action='store_true', help='use pre-trained model')

#Device options
parser.add_argument('--ngpu', default='1', type=int, help='Number of GPUs used')

"""Function used for Orthogonal Regularization"""

def l2_reg_ortho(model):
    l2_reg = None
    for W in model.parameters():      
        if W.ndimension() < 2:            
            continue
        else:
            
            cols = W[0].numel()
            rows = W.shape[0]
            w1 = W.view(-1,cols)
            wt = torch.transpose(w1,0,1)
            m  = torch.matmul(wt,w1)
            ident = Variable(torch.eye(cols,cols))
            ident = ident.cuda()

            w_tmp = (m - ident)
            height = w_tmp.size(0)
            u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
            v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
            u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
            sigma = torch.dot(u, torch.matmul(w_tmp, v))

            if l2_reg is None:
                l2_reg = (sigma)**2
            else:
                l2_reg = l2_reg + (sigma)**2
    return l2_reg


use_cuda = torch.cuda.is_available()
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

best_score = 0

def main():
    global best_score
    start_epoch = args.start_epoch
	
	#Data  Loader
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    
    data_train =datasets.ImageFolder(traindir,transforms.Compose([transforms.ToTensor()]))
    mean_tr, std_tr = get_mean_and_std(data_train)
    data_test =datasets.ImageFolder(valdir,transforms.Compose([transforms.ToTensor()]))
    mean_te, std_te = get_mean_and_std(data_test)
    
    #Note that for imgaug, we should convert the PIL images to NumPy arrays before applying the transforms.

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean_tr,std=std_tr)]))
    test_dataset = datasets.ImageFolder(valdir, transforms.Compose(
            [transforms.Resize(256),transforms.CenterCrop(224),	transforms.ToTensor(), transforms.Normalize(mean=mean_te,std=std_te)]))
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset  ,sampler=ImbalancedDatasetSampler(train_dataset)
            ,batch_size=args.train_batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_dataset #, sampler=ImbalancedDatasetSampler(test_dataset)
            ,batch_size=args.test_batch, shuffle=False,	num_workers=args.workers, pin_memory=True)
    
#    test_loader = torch.utils.data.DataLoader(test_dataset #, sampler=ImbalancedDatasetSampler(test_dataset)
#        ,batch_size=320, shuffle=False,	num_workers=args.workers, pin_memory=True)
    
#    for inputs, targets in train_loader:

	#Create Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet34()
    model = model.to(device)
    summary(model, (3,224,224))
#    for child in model.named_children():
#        print(child)
#    model.fc.weight
#    (list(model.layer4.children()))[0].conv1.weights
   

    #Get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    model = torch.nn.DataParallel(model).cuda() 
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name)
	#cudnn.benchmark = True
	
	# define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
				nesterov = args.nesterov,weight_decay=args.weight_decay)
    
    title = 'AF' 
    if args.resume:
        	# Load checkpoint.
    	print('==> Resuming from checkpoint..')
    	assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    	args.checkpoint = os.path.dirname(args.resume)
    	checkpoint = torch.load(args.resume)
    	best_score = checkpoint['best_score']
    	print(best_score)
    	start_epoch = checkpoint['epoch']
    	model.load_state_dict(checkpoint['state_dict'])
    	optimizer.load_state_dict(checkpoint['optimizer'])
    	logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc 1.', 'Valid Acc 1.'])
        
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return
	
	# Train and val
    for epoch in range(start_epoch, args.epochs):
    	adjust_learning_rate(optimizer, epoch)

#    	print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
    	#Adjust Orhto decay rate
    	odecay = adjust_ortho_decay_rate(epoch+1)
    	sendecay = adjust_sen_decay(epoch+1)
	
    	train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda,odecay,sendecay)
    	test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)
		
    	# append logger file
    	logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

    	# save model
    	is_best = test_acc > best_score
    	best_score = max(test_acc, best_score)
    	save_checkpoint({
            	'epoch': epoch + 1,
            	'state_dict': model.state_dict(),
            	'acc': test_acc,
            	'best_score': best_score,
            	'optimizer' : optimizer.state_dict(),
			}, is_best, checkpoint=args.checkpoint)
    	
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    print('Best Fscore:')
    print(best_score)

def train(train_loader, model, criterion, optimizer, epoch, use_cuda,odecay,sendecay):
	# switch to train mode
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
#    sencoe = torch.FloatTensor(33)
   
    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        
#        print(inputs[2].shape, targets[2].shape )
        # measure data loading time
        data_time.update(time.time() - end)
    
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            
        #l1-regularization
#        reg_loss = 0
#        l1_crit = nn.L1Loss(size_average=False)
#        for param in model.parameters():
#            reg_loss += l1_crit(param, torch.zeros_like(param))
#        factor = 0.0005
#        l1_loss = factor * reg_loss
#    
    	# compute output
        outputs = model(inputs)
        oloss = l2_reg_ortho(model)
        oloss = odecay * oloss
        loss = criterion(outputs, targets)
#        loss = loss  +l1_loss+ oloss 
   
        # Sensivitiy calcualtion
        Last_layer_grad = torch.autograd.grad(loss,outputs, retain_graph=True)
        sen_value = torch.mean(torch.abs(Last_layer_grad[0]))
#        sen_coef = torch.abs(Variable(sencoe, requires_grad=True))
#        clampped_sen_coef = torch.clamp(sen_coef, min=1, max=400)
#        sen_term = (10000*odecay*torch.clamp(loss, min=0 , max =1)) / (sen_value +0.01)
        sen_term = sendecay / (sen_value +0.0001) # Just use a multiplication of one as coefficient because it changes the gradients 
#        print(loss, oloss)
        
#        loss = (args.lambdaa*loss) + (1-args.lambdaa)*sen_term + oloss
#        loss = loss - sen_term
#        loss = loss  +((oloss + sen_term)/2)
        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

    
    	# compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    	# measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    	# plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                	batch=batch_idx + 1,
                	size=len(train_loader),
                	data=data_time.val,
                	bt=batch_time.val,
                	total=bar.elapsed_td,
                	eta=bar.eta_td,
                	loss=losses.avg,
                	top1=top1.avg,
                
                	)
        bar.next()
	
    bar.finish()
    return (losses.avg, top1.avg)
	
def test(val_loader, model, criterion, epoch, use_cuda):
	
    global best_score

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    Fscore = AverageMeter()
    
	# switch to evaluate mode
    model.eval()
    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
   
    for batch_idx, (inputs, targets) in enumerate(val_loader):
    # measure data loading time
    	data_time.update(time.time() - end)

    	if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
	#inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
    	with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
    
    # measure accuracy and record loss
    	outputs = model(inputs)
    	loss = criterion(outputs, targets)
    	prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
    	losses.update(loss.item(), inputs.size(0))
    	top1.update(prec1.item(), inputs.size(0))
    	
#    	a=precision_recall_fscore_support(targets.data, torch.max(outputs,1)[1], average='weighted')
#    	F_output =a[2]
#    	F_output = f1_score(targets.data, torch.max(outputs,1)[1], average="weighted")
#    	Fscore.update(F_output.item(), inputs.size(0))
        
	# measure elapsed time
    	batch_time.update(time.time() - end)
    	end= time.time()

	# plot progress
    	bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
            	batch=batch_idx + 1,size=len(val_loader), data=data_time.avg, bt=batch_time.avg,
            	total=bar.elapsed_td,eta=bar.eta_td,loss=losses.avg,
            	top1=top1.avg)
    	bar.next()
    
    bar.finish()
#    for batch_idx, (inputs, targets) in enumerate(test_loader):
#        
#        if use_cuda:
#            inputs, targets = inputs.cuda(), targets.cuda()
#        inputs = Variable(inputs, volatile=True)
#        targets = Variable(targets, volatile=True)
#        outputs = model(inputs)
##        target_names = ['class 0', 'class 1']
##        print(classification_report(targets.data, torch.max(outputs,1)[1], target_names=target_names)) 
#        print(f1_score(targets.data, torch.max(outputs,1)[1], average="weighted"))
#    print(Fscore.avg)      
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']

def adjust_ortho_decay_rate(epoch):
    o_d = args.ortho_decay

    if epoch > 160:
        o_d = 0.0
    elif epoch > 130:
    	o_d = 1e-2 * o_d    
    elif epoch > 90:
    	o_d = 1e-2 * o_d
    elif epoch > 40:
    	o_d = 1e-1 * o_d

    return o_d

def adjust_sen_decay(epoch):
    o_d = args.ortho_decay

    if epoch > 160:
        o_d = 0.0
    elif epoch > 130:
    	o_d = 1e-3 * o_d
    elif epoch > 90:
    	o_d = 1e-2 * o_d
    elif epoch > 40:
    	o_d = 1e-1 * o_d

    return o_d    
if __name__ == '__main__':
	main()
