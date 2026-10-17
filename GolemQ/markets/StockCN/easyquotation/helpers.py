# coding:utf8
import json
import os
import re
import requests

path = os.path.expanduser('~')
user_path = os.path.join(path, '.GolemQ')
DEFAULT_STOCK_CODE_PATH = os.path.join(os.path.dirname(__file__), "stock_codes.conf")
STOCK_CODE_PATH = os.path.join(user_path, 'datastore', 'cache', 'stock_cn', 'stock_codes.conf')
index_codes = [
    "sh000001", "sh000002", "sh000003", "sh000004", "sh000005", "sh000006", "sh000007", "sh000008", "sh000009", "sh000010", 
    "sh000011", "sh000012", "sh000015", "sh000016", "sh000017", "sh000018", "sh000019", "sh000020", "sh000021", "sh000022", 
    "sh000025", "sh000026", "sh000027", "sh000028", "sh000029", "sh000030", "sh000031", "sh000032", "sh000033", "sh000034", 
    "sh000035", "sh000036", "sh000037", "sh000038", "sh000039", "sh000040", "sh000041", "sh000042", "sh000043", "sh000044", 
    "sh000045", "sh000046", "sh000047", "sh000048", "sh000049", "sh000050", "sh000051", "sh000052", "sh000053", "sh000054", 
    "sh000055", "sh000056", "sh000057", "sh000058", "sh000059", "sh000060", "sh000061", "sh000062", "sh000063", "sh000064", 
    "sh000065", "sh000066", "sh000067", "sh000068", "sh000069", "sh000070", "sh000071", "sh000072", "sh000073", "sh000074", 
    "sh000075", "sh000076", "sh000077", "sh000078", "sh000079", "sh000090", "sh000091", "sh000092", "sh000093", "sh000094", 
    "sh000095", "sh000096", "sh000097", "sh000098", "sh000099", "sh000100", "sh000101", "sh000102", "sh000103", "sh000104", 
    "sh000105", "sh000106", "sh000107", "sh000108", "sh000109", "sh000110", "sh000111", "sh000112", "sh000113", "sh000114", 
    "sh000115", "sh000116", "sh000117", "sh000118", "sh000119", "sh000120", "sh000121", "sh000122", "sh000123", "sh000125", 
    "sh000126", "sh000128", "sh000129", "sh000130", "sh000131", "sh000132", "sh000133", "sh000134", "sh000135", "sh000136", 
    "sh000137", "sh000138", "sh000139", "sh000141", "sh000142", "sh000145", "sh000146", "sh000147", "sh000148", "sh000149", 
    "sh000150", "sh000151", "sh000152", "sh000153", "sh000155", "sh000158", "sh000159", "sh000160", "sh000161", "sh000162", 
    "sh000170", "sh000300", "sh000688", "sh000802", "sh000814", "sh000819", "sh000823", "sh000827", "sh000847", "sh000849", 
    "sh000851", "sh000852", "sh000853", "sh000854", "sh000855", "sh000856", "sh000857", "sh000858", "sh000863", "sh000865", 
    "sh000867", "sh000869", "sh000891", "sh000901", "sh000903", "sh000905", "sh000906", "sh000913", "sh000914", "sh000928", 
    "sh000932", "sh000933", "sh000934", "sh000935", "sh000974", "sh000982", "sh000986", "sh000987", "sh000989", "sh000991", 
    "sh000992", "sh000993", "zz000912", "zz000929", "zz000930", "zz000931", "zz000936", "zz000937", "zz000988", "zz000994", 
    "zz000995", "zz930606", "zz930697", "zz930910", "zz931008", "zz931009", "zz931160", "zz931719", "zz931775", "zz931892", 
    "zz931897", "zz931992"]


def save_stock_codes(codes):
    """把码表写进本地缓存 ``STOCK_CODE_PATH``（``{"stock": [...]}``）—— **唯一写口**。

    ⚠️ **写盘必须是原子的**：先写同目录的 ``.tmp``，再 ``os.replace`` 顶上去。
    读方（`basequotation.load_stock_codes`）读的是**同一个路径**，而本文件
    同时被每天 `--save tdx` 的落盘路径与 L1/L2 订阅器读 —— 写一半被中断
    （Ctrl-C、磁盘满、进程被杀）会留下半个 JSON，`use('tencent')` 当场抛，
    **两个订阅器一起起不来**。`os.replace` 在同一文件系统上是原子的。

    ⚠️ **父目录按需建**：``STOCK_CODE_PATH`` 在
    ``~/.GolemQ/datastore/cache/stock_cn/`` 下，全新机器上这几级**都不存在**。
    旧写法直接 ``open(..., 'w')`` 会 `FileNotFoundError` —— 只是它全树零调用
    （见 :func:`update_stock_codes`），所以从没暴露过。

    ``ensure_ascii=False`` 是为了让文件在编辑器里可读；码表本身是纯 ASCII，
    但别依赖这一点（`open` 一律显式带 `encoding`，见 CLAUDE.md 的编码纪律）。

    :param codes: 代码序列（6 位裸码或带 ``sh``/``sz``/``zz`` 前缀都行），会去重排序
    :returns: 实际写盘的有序列表
    """
    codes = sorted({str(c).strip() for c in codes if str(c).strip()})
    parent = os.path.dirname(STOCK_CODE_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = STOCK_CODE_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'stock': codes}, f, ensure_ascii=False)
    os.replace(tmp, STOCK_CODE_PATH)
    return codes


def update_stock_codes():
    """从 ``shdjt.com`` 抓全量码表 —— ⚠️ **本树已不再走这条路**（保留为上游 API）。

    实际生效的刷新在 `refdata_save.refresh_stock_codes_cache`：`--save tdx`
    每次都把 8.3 的 `stock_list`（pytdx 主源 + tdxaidata 补北交所）导出成码表。
    不在这里敲站点，是因为实测它**要 19.1 秒**（15s 超时下会偶发成功、偶发
    `ReadTimeout`）、且 ``https://`` 那道门直接拒绝连接 —— 拿它做每日刷新
    等于给 `--save tdx` 挂一个不稳定的外网依赖，而库里那份**刚刚刷过、更全**。

    写盘收敛到 :func:`save_stock_codes`（原子写 + 建父目录 + 显式编码）：
    这文件曾有**两份写法**，这一份既没建目录也没写编码 —— 平行实现不会报错，
    只会分叉。
    """
    response = requests.get("http://www.shdjt.com/js/lib/astock.js")
    stock_codes = re.findall(r"~([a-z0-9]*)`", response.text)
    return save_stock_codes(stock_codes)


def get_stock_codes(realtime=False):
    """获取所有股票 ID 到 all_stock_code 目录下"""
    if realtime:
        return update_stock_codes()
    if (os.path.exists(STOCK_CODE_PATH) and not os.path.isdir(STOCK_CODE_PATH)):
        with open(STOCK_CODE_PATH, encoding='utf-8') as f:
            return json.load(f)["stock"]
    else:
        with open(DEFAULT_STOCK_CODE_PATH, encoding='utf-8') as f:
            return json.load(f)["stock"]


def get_stock_type(stock_code):
    """判断股票ID对应的证券市场
    匹配规则
    ['50', '51', '60', '90', '110'] 为 sh
    ['00', '13', '18', '15', '16', '18', '20', '30', '39', '115'] 为 sz
    ['5', '6', '9'] 开头的为 sh， 其余为 sz
    :param stock_code:股票ID, 若以 'sz', 'sh' 开头直接返回对应类型，否则使用内置规则判断
    :return 'sh' or 'sz'"""
    assert type(stock_code) is str, "stock code need str type"
    sh_head = ("50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "90", "110", "113", "118",
               "132", "204", "5", "6", "9", "7")
    if stock_code.startswith(("sh", "sz", "zz")):
        return stock_code[:2]
    else:
        return "sh" if stock_code.startswith(sh_head) else "sz"
