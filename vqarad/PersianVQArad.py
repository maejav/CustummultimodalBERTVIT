
#### first read train.json
import json 
# from googletrans import Translator
from googletrans import Translator
import translators as ts

# wyw_text = '季姬寂，集鸡，鸡即棘鸡。棘鸡饥叽，季姬及箕稷济鸡。'
# chs_text = '季姬感到寂寞，罗集了一些鸡来养，鸡是那种出自荆棘丛中的野鸡。野鸡饿了唧唧叫，季姬就拿竹箕中的谷物喂鸡。'
# html_text = '''
# <!DOCTYPE html>
# <html>
# <head>
# 	<title>我是标题</title>
# </head>
# <body>
# <p>我是文章《你的父亲》</p>
# </body>
# </html>
# '''

# # input languages
# print(ts.google(wyw_text)) # default: from_language='auto', to_language='en'

# ## output language_map
# print(ts._google.language_map)

# # professional field
# print(ts.alibaba(wyw_text, professional_field='general')) # ("general","message","offer")
# print(ts.baidu(wyw_text, professional_field='common')) # ('common','medicine','electronics','mechanics')
# print(ts.caiyun(wyw_text, from_language='zh', professional_field=None)) # ("medicine","law","machinery")

# # property
# rs = [ts.tencent(x) for x in [wyw_text, chs_text]]
# print(ts._tencent.query_count)
# print(dir(ts._tencent))

# # requests
# print(ts.youdao(wyw_text, sleep_seconds=5, proxies={}))

# # host service
# print(ts.google(wyw_text, if_use_cn_host=True))
# print(ts.bing(wyw_text, if_use_cn_host=False))

# # detail result
# print(ts.sogou(wyw_text, is_detail_result=True))

# # translate html
# print(ts.translate_html(html_text, translator=ts.google, to_language='en', n_jobs=-1))

# # help
# help(ts.google)



f = open("datapersian.json")
data=json.load(f)
print(data)

# print("data in json file: ", data)
# print("type of data in json file: ", type(data)) ### json file is list of dictionary 



print("number of item in original json file: ",len(data))
# exit()

translator = Translator()

counter=0
new_data=[]
datapersion= {}
for i in range(0,len(data)):

    # input languages
    # print(ts.google(wyw_text)) # default: from_language='auto', to_language='en'

    ## output language_map
    # print(ts._google.language_map)
    # print(type(data[i]["sent"]))
    # print(data[i])
    # exit()
    try:
        Translator
        traslated_sentance= translator.translate(str(data[i]["question"]),dest="fa")
        translated_answer =translator.translate(str(data[i]["answer"]),dest="fa")
        # traslated_sentance= ts.google(data[i]["question"],to_laزnguage="fa")
        # translated_answer =ts.google(data[i]["answer"],to_language="fa")
        # print(traslated_sentance)
        # traslated_back_sentence= translator.translate(traslated_sentance,src="fr",dest="en").text
    

        # if data[i]["sent"]!=traslated_back_sentence:

            # new_dict=data[i]
            # new_dict["sent"]=traslated_back_sentence
        temp=data[i].copy()
        new_data.append(
                    {
                    "qid": temp["qid"],
                    "phrase_type": temp["phrase_type"],
                    "qid_linked_id": temp["qid_linked_id"],
                    "image_case_url": temp["image_case_url"],
                    "image_name": temp["image_name"],
                    "image_organ": temp["image_organ"],
                    "evaluation": temp["evaluation"],
                    "question": traslated_sentance,
                    "question_rephrase": temp["question_rephrase"],
                    "question_relation": temp["question_relation"],
                    "question_frame": temp["question_frame"],
                    "question_type": temp["question_type"],
                    "answer": translated_answer,
                    "answer_type": temp["answer_type"]

                    }
                )
            
            # print("the original sentence: ",data[i]["sent"])
            # print("the translated sentence: ", traslated_back_sentence)

        if i%5==0 and i!=0:
            data.extend(new_data)
            datapersion.extend(new_data)
            new_data=[]
            # with open('train_augmented.json', 'w') as outfile:
            #     json.dump(data, outfile)
                

            with open('datapersion2.json', 'w') as outfile:
                json.dump(datapersion, outfile)
                
            print(len(data))
        

    except AttributeError:
        print("can not do ")
    
    
    # print("the original sentence: ",data[i]["sent"])
    # print("the translated sentence: ", traslated_back_sentence)
    # if i==10:
    #     exit()
    



