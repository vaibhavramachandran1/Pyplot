
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pathlib as path

class PlotLogs():
    def __init__(self,log):
        self.log = log
        self.index = [] 
        self.acc = [] 
        self.loss = [] 
        self.raw = open(log,'r')
        lines = self.raw.read().splitlines()
        del lines[0]
        for i in lines:
            i = i.split('\t')
            self.index.append(int(i[0]))
            self.loss.append(format(float(i[1]),".2f"))
            self.acc.append(format(float(i[2]),".2f"))
        self.loss = [float(i) for i in self.loss]
        self.acc = [float(i) for i in self.acc]

    def get_val(self):
        return self.index, self.loss, self.acc


    def axis_def(self):
        self.x = self.index
        self.y1 = self.loss
        self.y2 = self.acc

    def split_plot(self):
        self.axis_def()
        # figure , axis = plt.subplots(ncols=2)
        fig = plt.figure(num=None,figsize=(10,5),dpi=100,facecolor='w',edgecolor='k')
        gs = gridspec.GridSpec(1, 2, figure=fig)
        
        axis = fig.add_subplot(gs[0,0])
        axis.plot(self.x, self.y1)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title("Epochs vs Loss")

        axis1 = fig.add_subplot(gs[0,1])
        axis1.plot(self.x, self.y2)
        axis1.set_xlabel("Epoch")
        axis1.set_ylabel("Accuracy")
        axis1.set_title("Epochs vs Accuracy")
        
        plt.tight_layout()
        plt.savefig(f'{self.log}.pdf')
        plt.show()

if __name__ == "__main__":
    src = "logs\resnet18"
    log = "\train.log"
    file = src + log;
    print(file)
    plt1 = PlotLogs("logs/resnet101_300ep/val.log")
    plt1.split_plot()
    # index, loss , acc = plt1.get_val()
    # loss = [float(i) for i in loss]
    # acc = [float(i) for i in acc]
    # plt.plot(index,loss)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()
