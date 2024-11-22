import json
import pandas as pd
import os


'''
处理数据集 得到uid_major
'''
class Data_split():
    def __init__(self,root,dealing,numer):
        self.root = root
        self.dealing = dealing
        self.numer = numer

    def main(self):
        with open(self.root + self.dealing + '.json', 'r', encoding='utf-8') as f:
            # content = ''
            line_count = 0
            for line in f:
                data = json.loads(line)
                line_count +=1
                if line_count >=self.numer:
                    break
                # print(data)
                # print(type(data))
                with open('data/practice_process.json', 'a') as f:
                    f.write(json.dumps(data)+'\n')

class Data_process_raw():
    def __init__(self,root,dealing):
        self.root = root
        self.dealing = dealing

    def main(self):
        re = []
        with open(self.root+self.dealing+'.json', 'r',encoding='utf-8') as f:
            for line in f:         #line :dict
                # print(type(line))
                # print(line)
                data = json.loads(line)       #data str
                # print(type(data))
                if data['id'] == '':
                    continue
                uid = data['id']
                if 'major' not in data or not data['major']:
                    continue
                major = data['major']
                re.append([uid, major])
        re = pd.DataFrame(re, columns=['uid', 'major'])
        re.to_csv('data/mid/uid_major.csv', index=0)


        re = []
        with open(self.root+self.dealing+'.json', 'r', encoding='utf-8') as f:
            for line in f:         #line :str
                data = json.loads(line)       #data dict
                if data['id'] == '':
                    continue
                uid = data['id']
                degree = data['degree']
                re.append([uid, degree])
        re = pd.DataFrame(re, columns=['uid', 'degree'])
        re.to_csv('data/mid/uid_degree.csv', index=0)

        re = []
        with open(self.root+self.dealing+'.json', 'r', encoding='utf-8') as f:
            for line in f:         #line :str
                data = json.loads(line)       #data dict
                if data['id'] == '':
                    continue
                uid = data['id']
                # if 'degree' not in data or not data['degree']:
                #     continue
                workExperienceList = data['workExperienceList']  #workExperienceList：list  每个list项目是一个dict
                for i in range(len(workExperienceList)):
                    re.append([uid, workExperienceList[i]["industry"]])
        re = pd.DataFrame(re, columns=['uid', 'industry'])
        re.to_csv('data/mid/uid_industry.csv', index=0)


        re = []
        with open(self.root+self.dealing+'.json', 'r', encoding='utf-8') as f:
            for line in f:         #line :str
                data = json.loads(line)       #data dict
                if data['id'] == '':
                    continue
                uid = data['id']
                # if 'degree' not in data or not data['degree']:
                #     continue
                workExperienceList = data['workExperienceList']  #workExperienceList：list  每个list项目是一个dict
                for i in range(len(workExperienceList)):
                    re.append([uid, workExperienceList[i]["position_name"]])
        re = pd.DataFrame(re, columns=['uid', 'position'])
        re.to_csv('data/mid/uid_position.csv', index=0)


        re = []
        with open(self.root+self.dealing+'.json', 'r', encoding='utf-8') as f:
            for line in f:         #line :str
                data = json.loads(line)       #data dict
                # uid = data['id']
                workExperienceList = data['workExperienceList']  #workExperienceList：list  每个list项目是一个dict
                for i in range(len(workExperienceList)):
                    re.append([workExperienceList[i]["industry"], workExperienceList[i]["position_name"]])
        re = pd.DataFrame(re, columns=['industry', 'position'])
        re.to_csv('data/mid/industry_position.csv', index=0)

class Data_process_raw_yd():
    def __init__(self,root,dealing):
        self.root = root
        self.dealing = dealing

    def main(self):
        re = []
        with open(self.root + self.dealing + '.json', 'r', encoding='utf-8') as f:
            for line in f:  # line :dict
                # data = json.loads(line)  # data str
                data = eval(line)
                if data['姓名'] == '':
                    continue
                uid = data['姓名']
                if '主修专业' not in data or not data['主修专业']:
                    continue
                major = data['主修专业']
                re.append([uid, major])
        re = pd.DataFrame(re, columns=['uid', 'major'])
        re.to_csv('data/mid/uid_major_yd.csv', index=0)

        re = []
        with open(self.root + self.dealing + '.json', 'r', encoding='utf-8') as f:
            for line in f:  # line :dict
                # data = json.loads(line)  # data str
                data = eval(line)
                if data['姓名'] == '':
                    continue
                uid = data['姓名']
                if '求职目标' not in data or not data['求职目标']:
                    continue
                position = data['求职目标']
                re.append([uid, position])
        re = pd.DataFrame(re, columns=['uid', 'position'])
        re.to_csv('data/mid/uid_position_yd.csv', index=0)

        re = []
        with open(self.root + self.dealing + '.json', 'r', encoding='utf-8') as f:
            for line in f:  # line :dict
                # data = json.loads(line)  # data str
                data = eval(line)
                if data['姓名'] == '':
                    continue
                uid = data['姓名']
                if '学历' not in data or not data['学历']:
                    continue
                degree_name = data['学历']
                if degree_name == '高中':
                    degree = 0
                elif degree_name == '大专' or degree_name == '中专':
                    degree = 1
                elif degree_name == '本科':
                    degree = 2
                else:
                    degree = 3
                re.append([uid, degree])
        re = pd.DataFrame(re, columns=['uid', 'degree'])
        re.to_csv('data/mid/uid_degree_yd.csv', index=0)

        re = []
        with open(self.root + self.dealing + '.json', 'r', encoding='utf-8') as f:
            for line in f:  # line :dict
                # data = json.loads(line)  # data str
                data = eval(line)
                if data['姓名'] == '':
                    continue
                uid = data['姓名']
                if '技能' not in data or not data['技能']:
                    continue
                skills = data['技能']
                for skill in skills:
                    re.append([uid, skill])
        re = pd.DataFrame(re, columns=['uid', 'skill'])
        re.to_csv('data/mid/uid_skill_yd.csv', index=0)


'''
处理数据集为数值形式
'''
class Data_process_mid():
    def __init__(self,root):
        self.root = root

    def read_mid(self,field0,field1,field2,field3,field4):
        path0 = self.root + 'mid/'+field0 +'.csv'
        re0 = pd.read_csv(path0)
        path1 = self.root + 'mid/'+field1 +'.csv'
        re1 = pd.read_csv(path1)
        path2 = self.root + 'mid/'+field2 +'.csv'
        re2 = pd.read_csv(path2)
        path3 = self.root + 'mid/'+field3 +'.csv'
        re3 = pd.read_csv(path3)
        path4 = self.root + 'mid/'+field4 +'.csv'
        re4 = pd.read_csv(path4,encoding='GBK')
        return re0,re1,re2,re3,re4

    def mapper(self,re0,re1,re2,re3,re4):     #re0:uid_degree  re1:uid_industry  re2:uid_major  re3:uid_position   re4:industry_position
        print('uid_degree,uid:{},degree:{}.'.format(len(set(re0.uid)),len(set(re0.degree))))
        print('uid_skill_yd,uid:{},skill:{}.'.format(len(set(re1.uid)),len(set(re1.skill))))
        print('uid_major,uid:{},major:{}.'.format(len(set(re2.uid)),len(set(re2.major))))
        print('uid_position,uid:{},position:{}.'.format(len(set(re3.uid)),len(set(re3.position))))
        print('positon_skill,position:{},skill:{}.'.format(len(set(re4.position)),len(set(re4.skill))))
        all_uid = set(re0.uid) | set(re1.uid) | set(re2.uid) | set(re3.uid)
        all_skill = set(re1.skill) | set(re4.skill)
        all_major = set(re2.major)
        all_position = set(re3.position) | set(re4.position)
        print("all_uid:{}".format(len(all_uid)))
        print("all_industry:{}".format(len(all_skill)))
        print("all_major:{}".format(len(all_major)))
        print("all_position:{}".format(len(all_position)))
        uid_dict = dict(zip(all_uid, range(len(all_uid))))
        # 将字典输出为 JSON 格式的字符串
        uid_dict_data = json.dumps(uid_dict)
        # 打开一个文件，并将 JSON 数据写入文件
        with open('data/uid_dict_data.json', 'w') as file:
            file.write(uid_dict_data)
        skill_dict = dict(zip(all_skill,range(len(all_skill))))
        major_dict = dict(zip(all_major,range(len(all_major))))
        position_dict = dict(zip(all_position,range(len(all_position))))
        # 将字典输出为 JSON 格式的字符串
        position_dict_data = json.dumps(position_dict)
        # 打开一个文件，并将 JSON 数据写入文件
        with open('data/position_dict_data.json', 'w') as file:
            file.write(position_dict_data)
        re0.uid = re0.uid.map(uid_dict)
        re1.uid = re1.uid.map(uid_dict)
        re1.skill = re1.skill.map(skill_dict)
        re2.uid = re2.uid.map(uid_dict)
        re2.major = re2.major.map(major_dict)
        re3.uid = re3.uid.map(uid_dict)
        re3.position = re3.position.map(position_dict)
        re4.skill = re4.skill.map(skill_dict)
        re4.position = re4.position.map(position_dict)
        return re0,re1,re2,re3,re4


    def save(self,re0,re1,re2,re3,re4):
        output_root = self.root+'ready'
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        print(output_root)
        re0.to_csv(output_root+'/uid_degree.csv',sep=',', header=None, index=False)
        re1.to_csv(output_root+'/uid_skill.csv',sep=',', header=None, index=False)
        re2.to_csv(output_root+'/uid_major.csv',sep=',', header=None, index=False)
        re3.to_csv(output_root+'/uid_position.csv',sep=',', header=None, index=False)
        re4.to_csv(output_root+'/position_skill.csv',sep=',', header=None, index=False)

    def main(self):
        re0, re1, re2, re3, re4 = self.read_mid(field0='uid_degree_yd',field1='uid_skill_yd',field2='uid_major_yd',field3='uid_position_yd',field4='position_skill')
        re0, re1, re2, re3, re4 = self.mapper(re0, re1, re2, re3, re4 )
        self.save(re0, re1, re2, re3, re4 )



if __name__ == "__main__":
    root = "./data/"
    dealing = "practice"    #_process
    # numer = 2
    # Data_process_raw_yd(root,'yd').main()      #加入移动的数据集
    # Data_split(root,dealing,numer).main()
    # dealing_process = "practice_process"
    # Data_process_raw(root,dealing_process).main()
    Data_process_mid(root).main()   #重新映射

