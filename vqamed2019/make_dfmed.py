import pandas as pd 
def make_df():

    ###  i got an error hear because i do not change here 
    # df_empty = pd.DataFrame(data=[["1","2","3","4","5","6","7"]], columns=\
    # ["rownum", "model_name","bert_model",\
    # "epoch","lr", "loss", "overalla_ccuracy"])
    df_empty = pd.DataFrame(columns = ["model_name","bert_model","image_embedding","epoch"\
    ,"lr", "loss_train", "overall_accuracy_train","loss_val","loss_test","overall_accuracy_test","all_test_bleu","category"])
    

    
    


    # df=df_empty.iloc[:, :-1]
    # df2=df.copy()
    # df_empty.to_excel("output_train.xlsx",sheet_name="sheet_train")
    
    # with pd.ExcelWriter('output.xlsx') as writer:  
    #     df.to_excel("output.xlsx",sheet_name="sheet_train")
    #     df2.to_excel(writer, sheet_name='Sheet_test')  
     
    df_empty.to_excel("output_train.xlsx", sheet_name="sheet_train")
    
    return df_empty
df =make_df()
