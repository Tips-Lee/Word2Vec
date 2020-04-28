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

4、训练数据加载
见 utils/data_utils.py

5、模型训练代码撰写
5.1 大的框架搭建
5.2 将各个部分代码完善
5.3 代码整理
5.4 运行命令 python train.py --data_path=./data/train.cbow.data --dict_path=./data/dictionary.json --network_name=w2w --embedding_size=128 --window_size=4 --CBOW_mean=True --structure=cbow --num_sampled=100
--batch_size=1500 --max_epoch=100 --optimizer_name=adam --learning_rate=0.001 --regularization=0.0001 --checkpoint_dir=./running/model/cbow --checkpoint_per_batch=100 --summary_dir=./running/graph/cbow