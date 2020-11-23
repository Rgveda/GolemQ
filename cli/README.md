# GolemQ

## 使用：

### 功能包括：

来源新浪财经的A股实盘行情 l1快照数据

在 repo 根目录下面，输入：

python -m GolemQ.cli --sub sina_l1

只保存部分股票数据，输入：

python -m GolemQ.cli --sub sina_l1 --codelist "600783、601069、002152、000582、002013、000960、000881
、000698、600742、600203、601186、601007、600328、600879"

### 读取实盘l1数据

这里完成的动作包括，读取l1数据，重采样为1min数据，重采样为日线/小时线数据，具体方法为打开 Jupyter，输入

*from GolemQ.fetch.kline import \(*
    *get_kline_price,*
    *get_kline_price_min,*
*\)*

*data_day, codename = get_kline_price\("600519", verbose=True\)*

为避免出现很多打印信息，可以设置参数 *verbose=False*

*data_day, codename = get_kline_price_min\("6003444", verbose=False\)*

### 已知Bug：

上证指数 000001 实盘走势和平安银行混淆。 目前已经修正 ——2020.11.22

成交量：Volumne和Amount 计算方式不对。

未能正确处理 *000001.XSHG 600519.SH* 这类格式的代码，能返回K线数据，但是不含今日实盘数据

### 常见问题

无法运行命令

*PS C:\Users\azai\source\repos\GolemQ> python -m GolemQ.cli --sub sina_l1*

提示

*C:\Users\azai\AppData\Local\Programs\Python\Python37\python.exe: Error 
while finding module specification for 'GolemQ.cli' 
(ModuleNotFoundError: No module named 'GolemQ')*

解决方法输入 cd .. 切换到上一层目录

*PS C:\Users\azai\source\repos\GolemQ> cd ..*

*PS C:\Users\azai\source\repos> python -m GolemQ.cli --sub sina_l1*

Program Last Time 3.762s

Not Trading time 现在是中国A股收盘时间 2020-10-15 16:28:05.310437

Not Trading time 现在是中国A股收盘时间 2020-10-15 16:28:07.314858

Not Trading time 现在是中国A股收盘时间 2020-10-15 16:28:09.323150

Not Trading time 现在是中国A股收盘时间 2020-10-15 16:28:11.334017

