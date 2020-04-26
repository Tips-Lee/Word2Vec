代码执行流程：
1、分词，结果保存磁盘
1.1、字典构建
2、根据CBOW 和 skip-gram 结构处理数据，并将数据保存磁盘
3、从磁盘加载数据，并提供一个按批次获取数据的API
4、网络结构
5、训练代码
6、应用代码和模型部署


1、文本数据分词
运行命令： python convert_data.py  --opt=split --split_input_file=./data/in_the_name_of_people.txt  --split_output_file=./data/cut_in_the_name_of_people.txt

2、 词典构建
运行命令  python convert_data.py  --opt=dictionary  --dict_input_file=./data/cut_in_the_name_of_people.txt  --dict_output_file=./data/dictionary.json  --dict_min_count=5

3、训练数据转换
构建cbow数据：
运行命令  python convert_data.py  --opt=record  --record_input_file=./data/cut_in_the_name_of_people.txt  --record_output_file=./data/train.cbow.data  --window=4 --structure=cbow  --allow_padding=True
构建skip-gram数据：
运行命令  python convert_data.py  --opt=record  --record_input_file=./data/cut_in_the_name_of_people.txt  --record_output_file=./data/train.skip-gram.data  --window=4 --structure=skip-gram  --allow_padding=False