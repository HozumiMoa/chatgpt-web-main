# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import re
import json
import time
import uuid
import queue
import random
import logging
import decimal
import requests
import datetime
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor

# python的第三方模块
from flask import Flask, request, Response, send_file

# 代码里面自己实现的模块
import get_logger
from gen_query import Gen_Model
from NW_ChatBot import NW_ChatBot
from chat_utils import Chat_Utils

# 示例化flask对象和初始化日志对象
app = Flask(__name__, static_folder='static')
get_logger.log_file(logging.INFO)

DEVICE_ID = "cuda:0"
chatbot = NW_ChatBot(DEVICE_ID)
gen_model = Gen_Model()

llm_queue = queue.Queue()
task_thread = ThreadPoolExecutor(max_workers=1)

with open("resource/requests.json", 'r', encoding='utf-8') as req_source:
    source = json.loads(req_source.read())
    major_url = source["major_url"]
    major_doc_url = source["major_doc_url"]
    

# 读取某目录下的子一级目录名
def get_sub_dirs(dir_path):
    sub_dirs = []
    try:
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isdir(file_path):
                sub_dirs.append(file_name)
    except:
        return []
    return sub_dirs


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        如有其他的需求可直接在下面添加
        :param obj:
        :return:
        """
        if isinstance(obj, datetime.datetime):
            # 格式化时间
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, datetime.date):
            # 格式化日期
            return obj.strftime('%Y-%m-%d')
        if isinstance(obj, decimal.Decimal):
            # 格式化高精度数字
            return float(obj)
        if isinstance(obj, uuid.UUID):
            # 格式化uuid
            return str(obj)
        if isinstance(obj, bytes):
            # 格式化字节数据
            return obj.decode("utf-8")
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


class FileChunkReader2:
    def __init__(self, file_path, chunk_size=1024):
        """
        初始化FileChunkReader类的实例。

        :param file_path: 文件的路径。
        :param chunk_size: 每个块的大小（以字节为单位）。
        """
        self.file_path = file_path
        self.chunk_size = chunk_size

    def iter_content(self, chunk_size):
        """从文件中迭代内容，返回每个内容块。"""
        with open(self.file_path, 'r') as file:
            while True:
                chunk = file.read(random.randint(1,2))
                if not chunk:
                    break
                result = f'"content":"{chunk}"'
                time.sleep(random.uniform(0, 0.05))
                yield result.encode('utf-8')
                
                
def hit_answer(hit_q, ori_q, templateCode):
    logging.info("—————命中回答—————")
    time.sleep(random.randint(2,3))
    start_time = datetime.datetime.now()
    hit_answer_path = f'/home/pipeline_server1213/mock/{hit_q}'
    os.makedirs(hit_answer_path, exist_ok=True)

    hit_response = FileChunkReader2(f'{hit_answer_path}/answer.txt', chunk_size=1024)
    hit_result = Chat_Utils.get_simple_response(query=ori_q, answer_source="GEN", answer=hit_response, file_source="制度")
    hit_result['templateCode'] = templateCode
    with open(f'{hit_answer_path}/slices.txt', 'r', encoding='utf-8') as f:
        json_content = f.read()

    try:
        slices = json.loads(json_content)
        if not isinstance(slices, list):
            slices = []
    except:
        slices = []
    hit_result['slices'] = slices[:10]
    try:
        with open(f'{hit_answer_path}/suggestion.txt', 'r', encoding='utf-8') as f:
            suggestions = f.read()
            if suggestions.strip() == '':
                suggestions = "相似问"
    except Exception as e:
        logging.error(f'读取相似问报错：{e}')
        suggestions = "相似问"
    hit_result['suggestions'] = "相似问"
    user_state= {
        "1": None,
        "2": None,
        "3": None,
        "4": None
    }
    result = Chat_Utils.nw_chat_utils(hit_result, start_time, "", 3, user_state)

    return result



def llm_task(queries, user_query="", images_dict={}):
    llm_thread_num = len(queries)
    if llm_thread_num == 0:
        llm_queue.put(None)
        return
    llm_thread = ThreadPoolExecutor(max_workers=llm_thread_num)
    # step1: 根据切片聚合数 submit到线程池
    future_list = []
    for query in queries:
        # 调用大模型
        future = llm_thread.submit(gen_model.req_llm, query, user_query, images_dict, llm_queue)
        future_list.append(future)
    #     llm_queue.put(future)
    
    # llm_queue.put(None)
    
    queue_task_counts = []
    while True:
        for i, future in enumerate(future_list):
            # step3: 谁先执行完，谁先put到队列中
            if not future.running() and i not in queue_task_counts:
                llm_queue.put(future)
                queue_task_counts.append(i)
                
        if len(queue_task_counts) == len(future_list):
            llm_queue.put(None)
            break

    # return future_list


@app.route('/customer_service', methods=['POST'])
def predict():
    """大模型入口"""
    try:
        """
        10000:系统错误
        10001:请求方法错误
        10002:请求参数错误或者参数格式类型错误
        10003:Token缺失
        10004:模型后端推理失败
        必填参数校验，参数接收之后
        1. path_id  int
        2. user_id  str
        3. query  str
        4. cids str 长度校验32位
        """
        total_start_time = time.time()
        if request.method != 'POST':
            return Response(json.dumps({"code": 10001, "error_msg": "请求方法错误"}))

        # 后台token_id
        token_id = None
        if "X-Access-Token" in request.headers:
            token_id = request.headers["X-Access-Token"]
            if not token_id:
                return Response(json.dumps({"code": 10003, "error_msg": "Token缺失"}))
            logging.info("token_id: {}".format(token_id))

        params = request if isinstance(request, dict) else request.json
        path_id = params.get('path_id', None)
        user_id = params.get('user_id', None)
        csid = params.get('csid', None)
        query = params.get('query', None)
        hist_list = params.get("history", [])
        user_state = params.get('user_state', {})
        #hist_list = params['history']
        #user_state = params['user_state']

        if not path_id or not isinstance(path_id, int):
            return Response(json.dumps({"code": 10002, "error_msg": "请求参数错误或者参数格式类型错误"}))
        if not user_id:
            return Response(json.dumps({"code": 10002, "error_msg": "请求参数错误或者参数格式类型错误"}))
        if not csid:
            return Response(json.dumps({"code": 10002, "error_msg": "请求参数错误或者参数格式类型错误"}))
        if not query or not isinstance(query, str):
            return Response(json.dumps({"code": 10002, "error_msg": "请求参数错误或者参数格式类型错误"}))

        logging.info("csid : {}".format(csid))
        logging.info("path id: {}".format(path_id))
        logging.info("user_id: {}".format(user_id))
        logging.info("query: {}".format(query))
        logging.info("hist_list: {}".format(hist_list))
        logging.info("user_state: {}".format(user_state))
        
        def generate2(response, query):
            """固定答案生成器"""
            iterator = response.pop("answer")  # 迭代器
            slices = response.pop("slices") # 切片
            for chunk in iterator.iter_content(1024):
                words = "".join(re.findall(r'"content":"(.*?)"',chunk.decode("utf-8","ignore"),re.DOTALL))
                #logging.info(f"words---->{words}")
                response["answer"] = words
                yield json.dumps(response, cls=JSONEncoder, ensure_ascii=False) + "<nw_dict>"

            response['slices'] = slices
            response['answer'] = ''
            yield json.dumps(response, cls=JSONEncoder, ensure_ascii=False) + "<nw_dict>"

        hit_q = query.replace("？", "")
        hit_q = hit_q.replace("?", "")
        hit_q = hit_q.replace("<nw_query_domain>", "=")
        templateCode = query.split('<nw_query_domain>')[1]

        hit_q_list = get_sub_dirs('/home/pipeline_server1213/mock')
        #logging.info(f'-------------{hit_q_list}--------:{hit_q_list}')
        if hit_q in hit_q_list:
            response_json = hit_answer(hit_q, query, templateCode)
            return Response(generate2(response_json, query))
        else:
            start_time = time.time()
            response_json = chatbot.nw_chat(path_id=path_id, query=query, hist_list=hist_list, user_state=user_state, token_id=token_id)
            logging.info(f"--------获取切片耗时----------：{time.time() - start_time}")

        # response_json = chatbot.nw_chat(path_id=path_id, query=query, hist_list=hist_list, user_state=user_state, token_id=token_id)
        #logging.info(f"Model prediction completed.{response_json}")
        queries = response_json.pop('queries', [])
        user_query = response_json.pop('user_query', "")
        domain = response_json.pop('domain', "")
        images_dict = response_json.pop('images_dict', {})
        # 开启多线程任务
        # task_thread.submit(llm_task, queries, user_query, images_dict)
            
        start_time1 = time.time()
        def generate(response):
            try:
                if not response["answer"]:
                    response['answer'] = "对不起，我没有在知识库中找到相关回答"
                    response['file_source'] = "制度"
                    response["suggestions"] = ""
                    response["warning"] = "此答案为AI生成，可能不准确，不代表南方电网公司立场或观点。"
                    yield json.dumps(response, cls=JSONEncoder, ensure_ascii=False) + "<nw_dict>"
                    # return Response(json.dumps({"code": -1, "error_msg": "当前资源不足, 请稍后重试."}))

                iterator = response.pop("answer")  # 迭代器
                is_iter = True if response["answer_source"] in ["GEN"] else False 
                response["templateCode"] = templateCode
                slices = response.pop('slices', [])
                slices = sorted(slices, key=lambda x: x['slice_similarity'][0], reverse=True)
                
                # 启动n个线程
                #threads = []
                #for thread_id, _query in enumerate(queries):
                #    thread = threading.Thread(target=gen_model.req_llm, args=(_query, user_query, domain, images_dict, llm_queue, thread_id))
                #    threads.append(thread)
                #    thread.start()
                
                thread_num = len(queries) if len(queries) else 1
                task_thread = ThreadPoolExecutor(max_workers=thread_num)
                future_list = []
                llm_start_time = time.time()
                for thread_id, _query in enumerate(queries):
                    future = task_thread.submit(gen_model.req_llm, _query, user_query, domain, images_dict, thread_id)
                    future_list.append(future)
                    llm_queue.put(future)
                    if thread_id == 0:
                        break
                    
                llm_queue.put(None)
                
                total_words = ""  # 文本写入docx
                logging.info("Iterating data begins")
                #while any(thread.is_alive for thread in threads) or not llm_queue.empty():
                while True:
                    try:
                        queue_result = llm_queue.get(timeout=0.1)
                        if queue_result is None:
                            break
                        logging.info(f'----------queue_result:{queue_result}')
                        future_result, thread_id = queue_result.result()
                        logging.info(f'----------线程id:{thread_id} 出结果了')
                        llm_response = future_result['answer']
                        file_source = future_result['file_source']
                        logging.info(f'-------llm_response: {llm_response}, {type(llm_response)}')
                        if isinstance(llm_response, str):
                            continue
                        else:
                            total_use_time = time.time() - total_start_time
                            logging.info(f'------从请求到第一个线程有结果的时间------:{total_use_time}')
                            current_words = ''
                            use_time = time.time() - start_time1
                            flag = True
                            #logging.info(f"----llm_start_time ---111111111111--：{time.time() - llm_start_time}")
                            for chunk in llm_response.iter_content(1024):
                                words = "".join(re.findall(r'"content":"(.*?)"',chunk.decode("utf-8","ignore"),re.DOTALL))
                                
                                #logging.info(f"words---->{words}")
                                
                                if words == 'あ':
                                    flag = False
                                    logging.info(f'---------特殊字符:{words}')
                                    continue
                                if flag:
                                    logging.info(f'-----第一个字出来的时间：{time.time() - total_start_time}')
                                    head_str = f"***库文件名***：{file_source}\n***问题答案***："
                                    response["answer"] = head_str
                                    response["file_source"] = "制度"
                                    yield json.dumps(response, cls=JSONEncoder, ensure_ascii=False) + "<nw_dict>"
                                    flag = False

                                # 替换图片链接
                                if words in images_dict:
                                    words = f'![]({images_dict[words]}.jpg)'
                                if words == '.jpg':
                                    words = ''

                                total_words += words
                                current_words += words
                                response["answer"] = words
                                response["file_source"] = "制度"
                                #logging.info(f'---------第一个响应 --------：{time.time() - total_start_time}')
                                yield json.dumps(response, cls=JSONEncoder, ensure_ascii=False) + "<nw_dict>"

                            if current_words:
                                logging.info(f'--------current_words-------:{current_words}')
                                response['answer'] = '\n\n\n'   # 分段流式间换行
                                response['file_source'] = "制度"
                                yield json.dumps(response, cls=JSONEncoder, ensure_ascii=False) + "<nw_dict>"

                            #use_time = time.time() - start_time1
                            #logging.info(f"二阶段流式耗时：{use_time}")
                    except queue.Empty:
                        logging.info('-------队列为空 跳过------')
                        # 检查线程是否都已完成
                        if not any(thread.is_alive() for thread in threads):
                            logging.info('-------所有线程已完成------')
                            break
                                
                if total_words == "":
                    response['answer'] = "对不起，我没有在知识库中找到相关回答"
                    response['file_source'] = "制度"
                    response["suggestions"] = ""
                    #response["warning"] = "此答案为AI生成，可能不准确，不代表南方电网公司立场或观点。"
                    yield json.dumps(response, cls=JSONEncoder, ensure_ascii=False) + "<nw_dict>"
                else:
                    response['slices'] = slices[:10]
                    response['answer'] = ''
                    response['file_source'] = "制度"
                    yield json.dumps(response, cls=JSONEncoder, ensure_ascii=False) + "<nw_dict>"

                logging.info("total words: {}".format(total_words))
                logging.info("Iteration data completed")
            except Exception as err:
                logging.error("Handle iteration process failure")
                logging.error(traceback.format_exc())
                yield json.dumps({"code": 10004, "error_msg": "模型后端推理失败"})

        return Response(generate(response_json))

    except Exception as err:
        logging.error(traceback.format_exc())
        return Response(json.dumps({"code": 10000, "error_msg": "系统错误"}))


# 心跳机制接口
@app.route('/health', methods=['GET'])
def health():
    result = {'status': 'UP'}
    return Response(json.dumps(result), mimetype='application/json')


@app.route('/get_image/<path:image_name>', methods=['GET'])
def get_static_file(image_name):
    try:
        logging.info(f"image_name:{image_name}")
        if(image_name.startswith('a')):
            logging.info("图片命中")
            return send_file(f"/home/pipeline_server1213/mockimage/{image_name}", as_attachment=True)
        else:
            return send_file(f"/home/download/{image_name}", as_attachment=True)

    except Exception as err:
        logging.error(traceback.format_exc())
        return Response(json.dumps({"code": "-1", "error_msg": "处理文件时出错."}), mimetype='application/json')


@app.route('/major/search', methods=['POST'])
def get_domain_majors():
    try:
        params = request if isinstance(request, dict) else request.json
        domain = params.get("domain", "")
        csid = params.get('csid', None)
        if not domain:
            return Response(json.dumps({"code": "-1", "error_msg": "缺少domain参数"}), mimetype='application/json')
        
        # 调es_server获取专业
        body = {"domain": domain, "csid": csid}
        res = requests.post(major_url, json=body, headers={"Content-Type": "application/json"}, verify=False)
        
        result = {"code": -1, "error_msg": "获取专业失败", "data": []}
        if res.status_code == 200:
            res_text = json.loads(res.text)
            if 'data' in res_text:
                if 'result' in res_text['data']:
                    response = res_text["data"]["result"]
                    result = {"code": 0, "msg": "获取专业成功", "data": response}

        return Response(json.dumps(result), mimetype='application/json')
    except Exception as err:
        logging.error(traceback.format_exc())
        return Response(json.dumps({"code": "-1", "error_msg": "获取专业时出错."}), mimetype='application/json')


@app.route('/major/doc/search', methods=['POST'])
def get_domain_major_doc():
    try:
        params = request if isinstance(request, dict) else request.json
        domain = params.get("domain", "")
        csid = params.get('csid', None)
        if not domain:
            return Response(json.dumps({"code": "-1", "error_msg": "缺少domain参数"}), mimetype='application/json')

        # 调es_server获取专业
        body = {"domain": domain, "csid": csid}
        res = requests.post(major_doc_url, json=body, headers={"Content-Type": "application/json"}, verify=False)

        result = {"code": -1, "error_msg": "获取专业下所有文件失败", "data": []}
        if res.status_code == 200:
            res_text = json.loads(res.text)
            if 'data' in res_text:
                if 'result' in res_text['data']:
                    response = res_text["data"]["result"]
                    result = {"code": 0, "msg": "获取专业成功", "data": response}

        return Response(json.dumps(result,ensure_ascii=False), mimetype='application/json')
    except Exception as err:
        logging.error(traceback.format_exc())
        return Response(json.dumps({"code": "-1", "error_msg": "获取专业时出错."}), mimetype='application/json')


@app.after_request
def after_request(response):
    """钩子后处理函数"""
    path = request.path
    if path == "/customer_service":
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Transfer-Encoding'] = 'chunked'
    else:
        response.headers['Content-Type'] = 'application/json'

    # # 添加缺少的安全HTTP头
    # response.headers.add('Content-Security-Policy', "default-src 'self'")
    # response.headers.add('X-Content-Type-Options', 'nosniff')
    # response.headers.add('X-XSS-Protection', '1; mode=block')
    # # 跨帧脚本编制防御:设置X-Frame-Options头来防止点击劫持攻击。
    # response.headers.add('X-Frame-Options', 'SAMEORIGIN')

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=63000, debug=False)
