
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss(df):

    colorr= ["green","blue","red"]
    linewid= [1,6,3]
    j=0

    for i in range(len(df)):
        epochs = df.loc[i,"epoch"]
        epoo = [epoch+1 for epoch in range(epochs)] 
        losses=df.loc[i,"loss"].strip('][').split(', ')
        losses=[float(loss) for loss in losses]
        stringlable1=str(df.loc[i,"model_name"])+ "_loss_plot"
        plt.plot(epoo, losses, color=colorr[j], label=stringlable1,linewidth=linewid[j])
        j+=1
        
    plt.legend()
    plt.savefig('comparemodelloss.png')

def plot_acc(df):
    
    colorr= ["green","blue","red"]
    linewid= [1,6,3]
    j=0
    for i in range(len(df)):
        epochs = df.loc[i,"epoch"]
        epoo = [epoch+1 for epoch in range(epochs)] 
        accs = df.loc[i,"overall_accuracy"].strip('][').split(', ')
        accs = [float(acc) for acc in accs]
        stringlabel2=str(df.loc[i, "model_name"]) + "_acc_plot"
        plt.plot(epoo, accs, color=colorr[j],\
        label=stringlabel2, linewidth=linewid[j])
        j+=1
        
    plt.legend()
    plt.savefig('comparemodeacc.png')
    




   

df = pd.read_excel("output_train.xlsx")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
plot_acc(df)
plt.clf() 
plot_loss(df)

