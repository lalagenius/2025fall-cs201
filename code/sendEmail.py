import pandas as pd

df = pd.read_excel('studentListInCourse-DSA-20250821.xls')
student_id = df['学号'].tolist()

sent_id = {

}

print(len(sent_id))

res1 = []
res2 = []
for i in student_id:
    if i in sent_id:
        continue

    res2.append(i)
    if i < 2e9:
        res1.append(f"{i}@pku.edu.cn")
    else:
        res1.append(f"{i}@stu.pku.edu.cn")

print(','.join(res1))
print(', '.join(map(str, res2)))

'''
我是mac机器，可以这样发送，但是北大邮箱总把收到的邮件归到Spam里面。
(cat mail.txt; uuencode 20230820-Wechat.jpg 20230820-Wechat.jpg)|mail -s "202308 计算概论（B）（12班）课程微信群" x@pku.edu.cn

所以我现在是程序输出要发送同学的邮箱地址列表，拷贝下来，在浏览器邮件里面发送。
'''
