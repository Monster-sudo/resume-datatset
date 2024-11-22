import math
from dateutil.relativedelta import relativedelta
import traceback
import json
import datetime

with open('./data/yd-copy.json', 'r', encoding='utf-8') as f:
    for line in f:  # line :dict
        # data = json.loads(line)  # data str
        res_data = eval(line)

        if not res_data['年龄']:
            print("###############################")
            print("简历中无年龄信息 需人工计算")
            print("###############################")
            if not res_data["出生日期"]:
                res_data['年龄'] = 0
            else:
                res_data['年龄'] = int(datetime.datetime.now().year) - int(res_data['出生日期'][:4]) + 1


        if "岁" in str(res_data["年龄"]):
            res_data["年龄"] = res_data["年龄"].split("岁")[0]
        res_data["年龄"] = int(res_data["年龄"])


        # 计算工作年限
        if res_data['工作时间'] == '':
            res_data['工作时间'] =0
        else:
            total_year = 0
            work_years = 0
            work_month = 0
            try:
                for work_time in res_data['工作时间']:
                    print(work_time)
                    if '-' in work_time and "至今" not in work_time:
                        time1, time2 = work_time.split('-')
                        # print(time1, time2)

                        # 定义起始日期和结束日期
                        start_date = datetime.datetime.strptime(time1, '%Y.%m')
                        end_date = datetime.datetime.strptime(time2, '%Y.%m')

                        # 计算工作年限
                        delta = relativedelta(end_date, start_date)

                        work_years += delta.years
                        work_month += delta.months
                        print("这段工作时间的年限为", work_years, work_month)

                        total_year += work_years
                    elif "至今" in work_time and "-" not in work_time:
                        time1 = work_time.split('至今')[0]
                        print("time1的时间为：", time1)

                        # 获取当前日期和时间
                        current_date = '2023.04'
                        now = datetime.datetime.strptime(current_date, '%Y.%m')
                        # now = datetime.datetime.now()

                        # 格式化为指定的数据格式
                        time2 = now.strftime("%Y.%m")

                        print("time2的时间为", time2)
                        # 定义起始日期和结束日期
                        start_date = datetime.datetime.strptime(time1, '%Y.%m')
                        end_date = datetime.datetime.strptime(time2, '%Y.%m')

                        # 计算工作年限
                        delta = relativedelta(end_date, start_date)
                        work_years += delta.years
                        work_month += delta.months
                        print("这段工作时间的年限为", work_years, work_month)
                        # total_year += work_years
                    elif "至今" in work_time and "-" in work_time:
                        time1 = work_time.split('-')[0]
                        print("time1的时间为：", time1)

                        # 获取当前日期和时间
                        # now = datetime.datetime.now()
                        current_date = '2023.04'
                        now = datetime.datetime.strptime(current_date, '%Y.%m')

                        # 格式化为指定的数据格式
                        time2 = now.strftime("%Y.%m")

                        # print("time2的时间为", time2 )
                        # 定义起始日期和结束日期
                        start_date = datetime.datetime.strptime(time1, '%Y.%m')
                        end_date = datetime.datetime.strptime(time2, '%Y.%m')

                        # 计算工作年限
                        delta = relativedelta(end_date, start_date)
                        work_years += delta.years
                        work_month += delta.months
                        print("这段工作时间的年限为", work_years, work_month)

                total_year = work_years + math.ceil(work_month / 12)
                print("{}的总的工作年限为{}".format(res_data['姓名'], total_year))
                res_data['工作时间'] = total_year
            except Exception as e:
                print("发生异常：", str(e))
                traceback.print_exc()
                # res_data['工作时间'] = 100





            if res_data['学历'] == '':
                res_data['学历'] = -1
            elif res_data['学历'] == '初中':
                res_data['学历'] = 0
            elif res_data['学历'] == '高中' :
                res_data['学历'] = 1
            elif res_data['学历'] == '中专' :
                res_data['学历'] = 2
            elif res_data['学历'] == '大专':
                res_data['学历'] = 3
            elif res_data['学历'] == '本科' :
                res_data['学历'] = 4
            elif res_data['学历'] == '硕士':
                res_data['学历'] = 5
            elif res_data['学历'] == '博士':
                res_data['学历'] = 6
            else:
                res_data['学历']= 7

        # degree_name = data['学历']
        # if degree_name == '高中':
        #     degree = 0
        # elif degree_name == '大专' or degree_name == '中专':
        #     degree = 1
        # elif degree_name == '本科':
        #     degree = 2
        # else:
        #     degree = 3

        with open('./data/yd-process.json', "a", encoding="utf-8") as f:
            json.dump(res_data, f, ensure_ascii=False)
            f.write('\n')
