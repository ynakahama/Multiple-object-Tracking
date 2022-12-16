import ast
import os

#画像の保管場所のパス変更
file_path_from = './pic/txt/'
file_path_to = './pic/txt_to/'

width_x, width_y = (720, 360)


path_label = 'pic\label.txt'
f = open(path_label, 'r')
data = f.read()
f.close()

data_dict = ast.literal_eval(data)

print(type(data_dict))
#print(data_dict.keys())



DIR = file_path_from
photo_number = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
print(photo_number)

for i in range(0, photo_number):
    label = open(file_path_to + '%06d.txt' % i,'w')
    with open(file_path_from + '%06d.txt' % i,'r') as f:
        for line in f:
            line_list = line.split(" ")
            if float(line_list[1]) in (0.0,2.0):
                if line_list[0] in data_dict.keys():
                    label.write(str(data_dict[line_list[0]]) + " " + str(line_list[2]) + " " + str(line_list[3]) + " " + str(line_list[4]) + " " + str(line_list[5]))
                else:
                    label.write("1.0" + " " + str(line_list[2]) + " " + str(line_list[3]) + " " + str(line_list[4]) + " " + str(line_list[5]))
                label.write("\n")
    f.close()
    label.close()








