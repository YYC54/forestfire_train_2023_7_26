1、环境
  所需安装包在ai-lab.yml 里
  使用方法conda env create --file ai-lab.yml
2、run.sh文件配置方法
   需要根据情况修改cd /public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire 后的文件路径到代码存放的文件夹
   /public/home/lihf_hx/anaconda3/envs/ai-lab/bin/python修改成环境安装位置/bin/python
  /public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/output.log 修改成代码存放的文件夹，output.log不变
（在windows上修改后在服务器上运行有可能出现换行符错误，这是由于unix和win换行符不一致导致，运行dos2unix run.sh转换为unix格式后再运行）
3、主函数入口
  main.py中的主函数中time为起报时间，bound为起报时间起要预测的天数（能够拿到10日预报数据故默认设为10）
  手动运行方法：激活环境>运行python main.py yyyymmdd
  通过shell脚本运行方法：sh run.sh yyyymmdd 也可直接sh run.sh (这样的默认起报时为当前日期)
  
4、路径文件存放说明
  路径配置文件存放在configs.ini文件中。[BASE_PATH]中是数据的存放路径（根据实际情况修改，到要素的上一级即可）。
 [TRAIN_PATH]中是和训练预测有关的相关参数。
 TRAIN参数设置为1时，运行后会训练+预测。（一般设置为0，有训练好的模型保存，若无法使用就设置为1重新训练）；
 为0时，运行后只会预测。
 train_dataset_north和train_dataset_west为处理好的训练集（相对路径，不需要更改）
 result_png是模型训练时产生的相关图像的保存位置（相对路径，不需要更改）
 model_path是模型的保存位置（相对路径，不需要更改）
 result_path是预测结果的保存位置，也是相对路径，如果想要生成到其他位置可以修改。

5、各个文件用处说明
  main.py：主函数
  run.sh:运行shell脚本
  output.log：日志文件
  configs.ini:配置文件
  dataset:训练数据集存放文件夹
  result:结果存放文件夹
  lib:各个功能code存放
  save:模型保存位置以及训练时产生的相关图像的保存位置
 


