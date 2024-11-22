import argparse
import torch
import torch.nn.functional as F
from data_loader import load_data
from TAHIN import TAHIN
import json
from datetime import datetime, timedelta


def softmax(x):
    # 将输入张量作为输入，应用softmax函数
    softmax_x = F.softmax(x, dim=-1)
    return softmax_x


def match_job(gpu, dataset, batch, num_workers, path, in_size, out_size, num_heads, dropout, user_name):
    # step 1: Check device
    if gpu >= 0 and torch.cuda.is_available():
        device = "cuda:{}".format(gpu)
    else:
        device = "cpu"

    # step 2: Load data
    (
        g,
        train_loader,
        eval_loader,
        test_loader,
        meta_paths,  # 元路径 对于Amazon数据集    meta_paths = {
        #     "user": [["ui", "iu"]],
        #     "item": [["iu", "ui"], ["ic", "ci"], ["ib", "bi"], ["iv", "vi"]],
        # }
        user_key,
        item_key,
    ) = load_data(dataset, batch, num_workers, path)  # Amazon数据集 batch
    g = g.to(device)  # 把图放到cuda中
    # print("Data loaded.")

    # step 3: Create model and training components
    model = TAHIN(  # 模型就是TAHIN
        g, meta_paths, in_size, out_size, num_heads, dropout
    )
    # summary(model,input_size=[(128,128)])
    model = model.to(device)  # 把模型放到cuda中进行训练
    # print("Model created.")

    model.eval()
    with torch.no_grad():
        model.load_state_dict(torch.load("TAHIN" + "_" + dataset))
        # 打开 JSON 文件并读取数据
        with open('data/uid_dict_data.json', 'r') as file:
            json_data = file.read()
        # 将 JSON 数据转换为字典
        uid_dict = json.loads(json_data)
        # 打开 JSON 文件并读取数据
        with open('data/position_dict_data.json', 'r') as file:
            json_data = file.read()
        # 将 JSON 数据转换为字典
        position_dict = json.loads(json_data)
        # user_name = '刘姿婷'     #输入用户名字（值）
        user_dict_key = uid_dict[user_name]  # 找到对应的键
        user = torch.tensor([user_dict_key])  # 将其转换成tensor 代入模型
        # item_name = '市场专员' #市场专员:478    市场总监:37
        # item_dict_key = position_dict[item_name]
        # print(item_dict_key)
        # item = torch.tensor([item_dict_key])
        item = torch.tensor([57,33,59,54,13,34,9,39,49,7])  # 输入职位的键  0 1 2     23
        logits1 = model.forward(g, user_key, item_key, user, item)  # 用户对职位的匹配度
        # result = find_key_by_value(position_dict,48)
        # print(result)
        # print('logits1:')
        sorted_tensor, indices = torch.sort(logits1 , descending=True)    #得到从大到小排序的匹配度和对应的职位的索引
        item = item.to(device)
        indices = indices.to(device)
        item_id = torch.index_select(item, dim=0, index=indices)
        # print("sorted_tensor")
        # print(sorted_tensor)
        # print("indices")
        # print(sorted_tensor, user, item_id)
        item_idd = item_id.tolist()
        j = 0
        job_set = []
        new_dict = {}
        for i in item_idd:
            pkeys = [key for key, val in position_dict.items() if val == i]
            if sorted_tensor[j] >= 0:
                print("用户{}".format(user_name) + "对于职位{}".format(pkeys) + '的适配为{}'.format(sorted_tensor[j]))
                new_dict[pkeys[0]] = float(sorted_tensor[j])
                job_set.append(pkeys)
            j = j + 1
        print(new_dict)
        # print(job_set)
        # flat_list = [item for sublist in job_set for item in sublist]
    return new_dict

def find_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def judge(user, job_name):  # user是一个字典 包含他的姓名，年龄，工作时间
    job = job_name
    if job_name == '产品运营':
        if user['主修专业'] == "市场营销":
            if user["工作时间"] >= 2:
                job = "产品运营"
            else:
                job = None
                print("抱歉，您没有2年及以上产品运营经验")
        else:
            job = None
            print("抱歉，您没有2年及以上产品运营经验")
    elif job_name == "设计师":
        if user['主修专业'] == '设计':
            if user["工作时间"] >= 2 and user["学历"] >= 3:
                job = "设计师"
            print("抱歉，您的主修专业没有大专以上或者没有两年相关经验，无法胜任设计师一职")
        else:
            job = None
            print("抱歉，您的主修专业没有大专以上或者没有两年相关经验，无法胜任设计师一职")
    elif job_name == "财务":
        if user['主修专业'] == "会计学":
            if user["学历"] >= 4:
                job = "财务"
            else:
                job = None
                print("主修专业不够，无法胜任财务")
        else:
            job = None
    elif job_name == "市场营销":
        if user['主修专业'] == "市场营销":
            if 'EXCEL' in user["技能"] and 'WORD' in user['技能'] and 'PPT' in user['技能'] and user["学历"] >= 4 and user["工作时间"] >= 10:
                job = "市场营销"
            else:
                job = None
                print("与市场营销不匹配")
        else:
            job = None
    elif job_name == "项目主管":
        if user['主修专业'] == '经贸MBA':
            if 'EXCEL' in user["技能"] and 'WORD' in user['技能'] and 'PPT' in user['技能'] and user["学历"] >= 4 and user["工作时间"] >= 3:
                job = "项目主管"
            else:
                job = None
                print("与项目主管不匹配")
        else:
            job = None
    elif job_name == "开发工程师":
        if user['主修专业'] == '计算机科学':
            if 'JAVA' in user["技能"] and user["学历"] >= 4 and user["工作时间"] >= 3:
                job = "开发工程师"
            else:
                job = None
                print("与开发工程师不匹配")
        else:
            job = None
    elif job_name == "文员":
        if user['主修专业'] == '商务英语':
            if user['年龄'] >= 25 and user['工作时间'] >= 1 and user["学历"] >= 3:
                job = "文员"
            else:
                job = None
                print("与文员不匹配")
        else:
            job = None
    elif job_name == "电商运营":
        if user['主修专业']== '市场营销' or user['主修专业'] == '运营管理':
            if user['工作时间'] >= 2:
                job = "电商运营"
            else:
                job = None
                print("与电商运营不匹配")
        else:
            job = None
    elif job_name == "人力资源管理":
        if user['主修专业'] == '人力资源管理' :
            if user['学历'] >= 3 and user["工作时间"] >= 2:
                job = "人力资源管理"
            else:
                job = None
                print("与人力资源管理不匹配")
        else:
            job = None
    elif job_name == "风控专员":
        if 'EXCEL' in user['技能'] and user['主修专业'] == '金融' and user['学历'] >= 5 and user['工作时间'] >= 5:
            job = "风控专员"
        elif 'EXCEL' in user['技能'] and user['主修专业'] == '国际贸易' and user['学历'] >= 5 and user['工作时间'] >= 5:
            job = "风控专员"
        else:
            job = None
            print("与风控专员不匹配")
    return job


def calculate_work_duration(start_date, end_date):
    current_date = '2023.07'
    start_date_obj = datetime.strptime(start_date, '%Y.%m')
    if end_date == '至今':
        end_date_obj = datetime.strptime(current_date, '%Y.%m')
    else:
        end_date_obj = datetime.strptime(end_date, '%Y.%m')
    duration = end_date_obj - start_date_obj
    print(duration)
    print(type(duration))
    return duration

def calculate_worktime(total_duration):
    years = total_duration.days / 365.25
    formatted_years = "{:.2f}".format(years)
    return formatted_years



if __name__ == "__main__":
    #读取一个json文件 一行一行读取
    with open('./data/yd.json', 'r', encoding='utf-8') as f:
        for line in f:  # line :dict
            # data = json.loads(line)  # data str
            res_data = eval(line)
            user = res_data

    # # 一个示例用户
    # user = {"user_name": '傅智翔', "age":25 , "工作时间":11, "学历":2 ,'技能':['EXCEL', 'WORD', 'PPT', 'JAVA'] , "主修专业":"市场营销" }
    #
    # # total_duration = sum((calculate_work_duration(exp.split('-')[0], exp.split('-')[1])).days for exp in user["work_time"])
    # # total_duration = timedelta(days=total_duration)
    # # work_years = calculate_worktime(total_duration)
    # # user["work_time"] = float(work_years)

    #进行工作匹配，得到一个工作列表
            if user['姓名'] == '':
                new_dict = {}
            else:
                new_dict = match_job(gpu=0, dataset="jobmatch", batch=4, num_workers=4, path="./data", in_size=64, out_size=64, num_heads=4,
                          dropout=0.1, user_name=user['姓名'])
                new_dict['姓名'] = user['姓名']
                # 筛选掉不符合条件的推荐岗位
                # job_sett = []
                # result = []
                # for job in job_set:
                #     job = judge(user,job)
                #     job_sett.append(job)
                #     result = list(filter(lambda x: x is not None, job_sett))
                # print("经过岗位要求具体筛选后：")
                # print(result)
                # print(type(result))
                # if not result:
                #     user['推荐岗位'] = ''
                # else:
                #     user['推荐岗位'] = result[0]
            with open('./data/yd-process--2.json', "a", encoding="utf-8") as f:
                # json.dump(new_dict, f, ensure_ascii=False)
                json.dumps(new_dict)
                json.dump(new_dict, f, ensure_ascii=False)
                # json_str = json.dumps(my_dict)
                f.write('\n')



# degree_name = data['学历']
# if degree_name == '高中':
#     degree = 0
# elif degree_name == '大专' or degree_name == '中专':
#     degree = 1
# elif degree_name == '本科':
#     degree = 2
# else:
#     degree = 3