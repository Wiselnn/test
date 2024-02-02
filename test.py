from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, GenerationConfig
import io
import json
import os
import random
import re
import torch
# from petrel_client.client import Client
from tqdm import tqdm


def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decoding error in file {file_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def read_jsonl(file_path):
    data = []
    def print_func(s: str) -> bool:
	print(s)
	return True
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 读取每一行并将其解析为JSON对象
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

# # new7b
# ckpt_path = '/mnt/petrelfs/share_data/dongxiaoyi/share_models/new7B_SFT'
# print(ckpt_path)
# # srun 
# # load model
# tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     ckpt_path, device_map='cuda', trust_remote_code=True).half().eval()
# device = torch.device('cuda', 0)
# torch.manual_seed(12345)

# /mnt/petrelfs/weixilin/.cache/huggingface/hub/models--baichuan-inc--Baichuan-7B/snapshots/c1a5c7d5b7f50ecc51bb0e08150a9f12e5656756
# /mnt/petrelfs/weixilin/.cache/huggingface/hub/models--THUDM--chatglm3-6b/chatglm3-6b

model = AutoModel.from_pretrained("/mnt/petrelfs/weixilin/.cache/huggingface/hub/models--THUDM--chatglm3-6b/chatglm3-6b", trust_remote_code=True, device='cuda')
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-14B-Chat", device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/mnt/petrelfs/weixilin/.cache/huggingface/hub/models--THUDM--chatglm3-6b/chatglm3-6b", trust_remote_code=True, device='cuda')
# model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()


# tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-13B-Chat", use_fast=False, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-13B-Chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
# model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-13B-Chat")

root_path = '/mnt/petrelfs/share_data/zhangpan/mllm/llm_data/v001-longyuan'
data_list = os.listdir('/mnt/petrelfs/share_data/zhangpan/mllm/llm_data/v001-longyuan')
def call_glm(text):
    global model
    model = model.eval()
    response, history = model.chat(tokenizer, text, history=[])
    # print(history)
    # print(response)
    return response

def call_new7b(text):
    meta_instruction = """<|System|>:You are an AI assistant whose name is InternLM (书生·浦语).\n- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.<eosys>\n """  # noqa: E501
    import json
    global model, tokenizer
    def gen_artical(text, temp=1., rept=1.1, sample=True, model=model, tokenizer=tokenizer):
        text = f'<bos>{meta_instruction}<|Human|>:{text} <|Bot|>:'
        print(text)
        input_ids = tokenizer(text, return_tensors='pt')['input_ids']
        with torch.no_grad():
            generate = model.generate(
                input_ids.cuda(),
                do_sample=sample,
                temperature=temp,
                repetition_penalty=rept,
                max_new_tokens=max(1024, 1000),
                top_p=0.8,
                top_k=40,
                length_penalty=1.0)
        response = tokenizer.decode(generate[0].tolist(), skip_special_tokens=True)
        return response

    gen_artical(text)
    
def call_baichuan13b(text):
    messages = []
    messages.append({"role": "user", "content": text})
    response = model.chat(tokenizer, messages)
    return response

def call_baichuan(text):
    global model, tokenizer
    inputs = tokenizer(text, return_tensors='pt')
    inputs = inputs.to('cuda')
    pred = model.generate(**inputs, max_new_tokens=4096,repetition_penalty=1.1)
    return tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)

def call_qwen(text):
    global model, tokenizer
    response, history = model.chat(tokenizer, text, history=None)
    return response

def call_test(text):
    global model
    model.eval()
    history = [{'role': 'user', 'content': """
你是一个学术论文分类器，你的任务是根据学术论文的特点，判断给定文章是否为学术论文。

学术论文特点：
学术论文是对某个科学领域中的学术问题进行研究后表述科学研究成果的理论文章。
学术论文应具备以下四个特点：
      一、科学性。学术论文的科学性，要求作者在立论上不得带有个人好恶的偏见，不得主观臆造，必须切实地从客观实际出发，从中引出符合实际的结论。在论据上，应尽可能多地占有资料，以最充分的、确凿有力的论据作为立论的依据。在论证时，必须经过周密的思考，进行严谨的论证。
      二、创造性。科学研究是对新知识的探求。创造性是科学研究的生命。学术论文的创造性在于作者要有自己独到的见解，能提出新的观点、新的理论。这是因为科学的本性就是“革命的和非正统的”，“科学方法主要是发现新现象、制定新理论的一种手段，旧的科学理论就必然会不断地为新理论推翻。”（斯蒂芬·梅森）因此，没有创造性，学术论文就没有科学价值。
      三、理论性。学术论文在形式上是属于议论文的，但它与一般议论文不同，它必须是有自己的理论系统的，不能只是材料的罗列，应对大量的事实、材料进行分析、研究，使感性认识上升到理性认识。一般来说，学术论文具有论证色彩，或具有论辩色彩。论文的内容必须符合历史唯物主义和唯物辩证法，符合“实事求是”、“有的放矢”、“既分析又综合” 的科学研究方法。
      四、平易性。指的是要用通俗易懂的语言表述科学道理，不仅要做到文从字顺，而且要准确、鲜明、和谐、力求生动。
    """.strip()}, 
    {'role': 'assistant', 'content': '好的，那我们开始吧！'},
    {'role': 'user', 'content': """
摘要：全民健身热潮不断，新媒体也在互联网技术的逐渐醇熟后迅速成长，两个行业的交汇成为了流行，掌播体育以此为背景，立足于为大众体育服务的新媒体组织定位，根据客户不同需求以不同资源链构成业务系统，以独特的切入点开辟了体育+新媒体的新道路。本研究旨在运用文献法、逻辑分析法、访谈法等研究方法，对掌播体育的商业模式进行整理和分析，批判性地反思其优点以及商业模式的掣肘，为从业者与相关研究者提供案例参考。

关键词：体育产业；新媒体；掌播体育；体育传媒；商业模式

一、掌播体育的发展背景

为进一步加快发展体育产业，促进体育消费，2014年10月，国务院颁布《关于加快发展体育产业促进体育消费的若干意见》文件...
    """.strip()
    },
    {'role': 'assistant', 'content': """
判断：该文章是学术论文。
理由：因为它具有学术论文基本的格式（摘要、关键词、正文等），同时它通过文献法、逻辑分析法、访谈法等研究方法对掌播体育商业模式进行整理和分析，具备科学性、创造性、理论性，并以通俗易懂的语言表述科学道理。
    """.strip()},
    {'role': 'user', 'content': """
《科技创新导报》经国家科技部和国家新闻出版总署批准公开发行的国家级综合科技期刊。主管单位：中国航天科技集团公司，主办单位：中国宇航出版有限责任公司。国内统一刊号：CN11-5640/N；国际标准刊号：ISSN1674-098X。邮发代号：80-542。国际大16开本精美印刷。

本刊已被《中国核心期刊（遴选）数据库》《中国学术期刊（光盘版）》《万方数据数字化期刊群》《中文科技期刊数据库》等网络媒体收录。

为促进专利成果转化，对现有的专利成果进行宣传推广，本刊特新增“专利推介”栏目，旨在对国内的专利成果进行学术推广，为专利发明者和专利使用者提供学术交流平台。现本栏目正式面向社会征稿。优惠优先发...
    """.strip()
},
{'role': 'assistant', 'content':"""
判断：该文章不是学术论文。
理由：该文章更像是一份刊物的介绍，提到了《科技创新导报》的主管单位、主办单位、刊号等信息，但未涉及对某个科学领域中学术问题的研究或科学研究成果的表述。学术论文应该具备对科学问题的研究和理论成果的表述，而这份介绍更偏向期刊自身的信息宣传。
""".strip()
}
    ]
    response, history = model.chat(tokenizer, text, history)
    # print(history)
    # print('-------------------------------')
    return response
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)

# client = Client()
# root_path = "p_ceph:s3://llm-release/release-231130/filtered/train/zh/zh-ebook-literature/"
# data_list = sorted(list(client.list(root_path)))

prompt = f"""
你是一个文章分类器，你的任务是根据不同文章的特点，将其划分到对应的类别。（注意：划分类别需要在给定类别集合中）
给定类别集合
```
学术论文 新闻稿 古诗词 现代诗歌 散文 小说
```
给定文章内容
```
\[content]
```
根据给定的文章内容，我将这篇文章归类为：
""".strip()

xslw = f"""
你是一个学术论文分类器，你的任务是根据学术论文的特点，判断给定文章是否为学术论文。

学术论文特点：
学术论文是对某个科学领域中的学术问题进行研究后表述科学研究成果的理论文章。
学术论文应具备以下四个特点：
一、科学性。学术论文的科学性，要求作者在立论上不得带有个人好恶的偏见，不得主观臆造，必须切实地从客观实际出发，从中引出符合实际的结论。在论据上，应尽可能多地占有资料，以最充分的、确凿有力的论据作为立论的依据。在论证时，必须经过周密的思考，进行严谨的论证。
二、创造性。科学研究是对新知识的探求。创造性是科学研究的生命。学术论文的创造性在于作者要有自己独到的见解，能提出新的观点、新的理论。这是因为科学的本性就是“革命的和非正统的”，“科学方法主要是发现新现象、制定新理论的一种手段，旧的科学理论就必然会不断地为新理论推翻。”（斯蒂芬·梅森）因此，没有创造性，学术论文就没有科学价值。
三、理论性。学术论文在形式上是属于议论文的，但它与一般议论文不同，它必须是有自己的理论系统的，不能只是材料的罗列，应对大量的事实、材料进行分析、研究，使感性认识上升到理性认识。一般来说，学术论文具有论证色彩，或具有论辩色彩。论文的内容必须符合历史唯物主义和唯物辩证法，符合“实事求是”、“有的放矢”、“既分析又综合” 的科学研究方法。
四、平易性。指的是要用通俗易懂的语言表述科学道理，不仅要做到文从字顺，而且要准确、鲜明、和谐、力求生动。

示例一：
```
摘要：全民健身热潮不断，新媒体也在互联网技术的逐渐醇熟后迅速成长，两个行业的交汇成为了流行，掌播体育以此为背景，立足于为大众体育服务的新媒体组织定位，根据客户不同需求以不同资源链构成业务系统，以独特的切入点开辟了体育+新媒体的新道路。本研究旨在运用文献法、逻辑分析法、访谈法等研究方法，对掌播体育的商业模式进行整理和分析，批判性地反思其优点以及商业模式的掣肘，为从业者与相关研究者提供案例参考。

关键词：体育产业；新媒体；掌播体育；体育传媒；商业模式

一、掌播体育的发展背景

为进一步加快发展体育产业，促进体育消费，2014年10月，国务院颁布《关于加快发展体育产业促进体育消费的若干意见》文件
```
判断：该文章是学术论文。
理由：因为它具有学术论文基本的格式（摘要、关键词、正文等），同时它通过文献法、逻辑分析法、访谈法等研究方法对掌播体育商业模式进行整理和分析，具备科学性、创造性、理论性，并以通俗易懂的语言表述科学道理。

示例二：
```
《科技创新导报》经国家科技部和国家新闻出版总署批准公开发行的国家级综合科技期刊。主管单位：中国航天科技集团公司，主办单位：中国宇航出版有限责任公司。国内统一刊号：CN11-5640/N；国际标准刊号：ISSN1674-098X。邮发代号：80-542。国际大16开本精美印刷。

本刊已被《中国核心期刊（遴选）数据库》《中国学术期刊（光盘版）》《万方数据数字化期刊群》《中文科技期刊数据库》等网络媒体收录。

为促进专利成果转化，对现有的专利成果进行宣传推广，本刊特新增“专利推介”栏目，旨在对国内的专利成果进行学术推广，为专利发明者和专利使用者提供学术交流平台。现本栏目正式面向社会征稿。优惠优先发
```
判断：该文章不是学术论文。
理由：该文章更像是一份刊物的介绍，提到了《科技创新导报》的主管单位、主办单位、刊号等信息，但未涉及对某个科学领域中学术问题的研究或科学研究成果的表述。学术论文应该具备对科学问题的研究和理论成果的表述，而这份介绍更偏向期刊自身的信息宣传。

```
/[content]
```
""".strip()

def test():
    root_path = '/mnt/petrelfs/share_data/zhangpan/mllm/llm_data/v001-longyuan/'
    data = sorted(os.listdir('/mnt/petrelfs/share_data/zhangpan/mllm/llm_data/v001-longyuan'))[0]
    data_list = read_jsonl(root_path+data)
    for d in data_list:
        print(f'{d["content"][:8093]}\n {call_test(d["content"][:8093])}')
        # break
        # print('----------------------------------')


# prompt = f"""
# 文章内容：
# ```
# \[content]
# ```
# 给定风格：
# ```
# 体裁
#   - 学习类：学术论文，议论文，作业（学习心得，读后感），作文，教科书
#   - 求学/求职类：自我介绍，求职信，个人陈述，个人简历
#   - 工作类：沟通邮件，通知邮件，OKR文档，技术文档，法律文书，合同文书，工作报告
#   - 营销类：产品介绍，活动通知，推文广告文案，营销软文，网店，带货文案
#   - 自媒体：自媒体文章（公众号，头条号，小红书），社交媒体（朋友圈，抖音快手短视频文案，直播稿，博客）
#   - 创作类：新闻稿，评价（影评书评），影视解说，攻略
#   - 文学类：小说，散文，现代诗歌，剧本，传记，杂文，歌词，古诗词，文言文，笑话
#   - 生活类：日记，游记，短信，情书
#   - 其他：观点看法，情感陪伴，角色扮演，建议指导，多轮对话
# 语气/文风
#   - 专业，口语，幽默，热情，礼貌，简洁，优雅，评论，讨论，提问，轻松，严肃 
#   - 亲切的，奢华的，轻松的，专业的，大胆的，敢于创新的，机智的，有说服力的，同感的
#   - 词藻华丽，直抒胸臆，平铺直叙，抒情，说教
# ```
# 请将上面文章内容，按照上述给定风格对本文的风格进行分类。
# 注意：最终输出需要从给定风格中选择，不能随意生成，允许多选，格式需要通过“、”隔开，不需要任何解释和总结，只给出分类结果即可
# 示例一：
# - 体裁：古诗词
# - 语气/文风：优雅、奢华的
# 示例二：
# - 体裁：求职信
# - 语气/文风：礼貌、严肃、简洁
# """.strip()

prompt1 = f"""
文章名：\[title]
所属杂志类别：\[zazhi]
文章内容：
```
\[content]
```
给定风格：
```
体裁
  - 学习类：学术论文，议论文，作业（学习心得，读后感），作文，教科书
  - 求学/求职类：自我介绍，求职信，个人陈述，个人简历
  - 工作类：沟通邮件，通知邮件，OKR文档，技术文档，法律文书，合同文书，工作报告
  - 营销类：产品介绍，活动通知，推文广告文案，营销软文，网店，带货文案
  - 自媒体：自媒体文章（公众号，头条号，小红书），社交媒体（朋友圈，抖音快手短视频文案，直播稿，博客）
  - 创作类：新闻稿，评价（影评书评），影视解说，攻略
  - 文学类：小说，散文，现代诗歌，剧本，传记，杂文，歌词，古诗词，文言文，笑话
  - 生活类：日记，游记，短信，情书
  - 其他：观点看法，情感陪伴，角色扮演，建议指导，多轮对话
语气/文风
  - 专业，口语，幽默，热情，礼貌，简洁，优雅，评论，讨论，提问，轻松，严肃 
  - 亲切的，奢华的，轻松的，专业的，大胆的，敢于创新的，机智的，有说服力的，同感的
  - 词藻华丽，直抒胸臆，平铺直叙，抒情，说教
```
请将上面文章内容，按照上述给定风格对本文的风格进行分类。
注意：最终输出需要从给定风格中选择，不能随意生成，允许多选，格式需要通过“、”隔开，不需要任何解释和总结，只给出分类结果即可
示例一：
- 体裁：古诗词
- 语气/文风：优雅、奢华的
示例二：
- 体裁：求职信
- 语气/文风：礼貌、严肃、简洁
""".strip()

prompt_zhihu = f"""
文章名：\[title]
文章内容：
```
\[content]
```
文章大纲：
```
\[outline]
```
给定风格：
```
体裁
  - 学习类：学术论文，议论文，作业（学习心得，读后感），作文，教科书
  - 求学/求职类：自我介绍，求职信，个人陈述，个人简历
  - 工作类：沟通邮件，通知邮件，OKR文档，技术文档，法律文书，合同文书，工作报告
  - 营销类：产品介绍，活动通知，推文广告文案，营销软文，网店，带货文案
  - 自媒体：自媒体文章（公众号，头条号，小红书），社交媒体（朋友圈，抖音快手短视频文案，直播稿，博客）
  - 创作类：新闻稿，评价（影评书评），影视解说，攻略
  - 文学类：小说，散文，现代诗歌，剧本，传记，杂文，歌词，古诗词，文言文，笑话
  - 生活类：日记，游记，短信，情书
  - 其他：观点看法，情感陪伴，角色扮演，建议指导，多轮对话
语气/文风
  - 专业，口语，幽默，热情，礼貌，简洁，优雅，评论，讨论，提问，轻松，严肃 
  - 亲切的，奢华的，轻松的，专业的，大胆的，敢于创新的，机智的，有说服力的，同感的
  - 词藻华丽，直抒胸臆，平铺直叙，抒情，说教
```
请将上面文章内容，按照上述给定风格对本文的风格进行分类。
注意：最终输出需要从给定风格中选择，不能随意生成，允许多选，格式需要通过“、”隔开，不需要任何解释和总结，只给出分类结果即可
示例一：
- 体裁：古诗词
- 语气/文风：优雅、奢华的
示例二：
- 体裁：求职信
- 语气/文风：礼貌、严肃、简洁
""".strip()


prompt_qwen = f"""
文章内容：
\[content]


给定风格：
体裁
  - 学习类：学术论文，议论文，作业（学习心得，读后感），作文，教科书
  - 求学/求职类：自我介绍，求职信，个人陈述，个人简历
  - 工作类：沟通邮件，通知邮件，OKR文档，技术文档，法律文书，合同文书，工作报告
  - 营销类：产品介绍，活动通知，推文广告文案，营销软文，网店，带货文案
  - 自媒体：自媒体文章（公众号，头条号，小红书），社交媒体（朋友圈，抖音快手短视频文案，直播稿，博客）
  - 创作类：新闻稿，评价（影评书评），影视解说，攻略
  - 文学类：小说，散文，现代诗歌，剧本，传记，杂文，歌词，古诗词，文言文，笑话
  - 生活类：日记，游记，短信，情书
  - 其他：观点看法，情感陪伴，角色扮演，建议指导，多轮对话
语气/文风
  - 专业，口语，幽默，热情，礼貌，简洁，优雅，评论，讨论，提问，轻松，严肃 
  - 亲切的，奢华的，轻松的，专业的，大胆的，敢于创新的，机智的，有说服力的，同感的
  - 词藻华丽，直抒胸臆，平铺直叙，抒情，说教


请将上面文章内容，按照上述给定风格对本文的风格进行分类。
注意：最终输出需要从给定风格中选择，不能随意生成，允许多选，格式需要通过“、”隔开，不需要任何解释和总结，只给出分类结果即可
示例一：
- 体裁：古诗词
- 语气/文风：优雅、奢华的
示例二：
- 体裁：求职信
- 语气/文风：礼貌、严肃、简洁
""".strip()

prompt_qwen2 = f"""
你是一个文章分类器，你的任务是根据不同文章的特点，将其划分到对应的类别。

给定文章内容:
\[content]

给定类别集合:
学术论文 新闻稿 古诗词 现代诗歌 散文 小说 其它

注意：划分类别需要在给定类别集合中！
请分类：
""".strip()


# s = []

# for data in tqdm(data_list):
#     cnt = client.get(root_path + data)
#     cnt = io.BytesIO(cnt)
#     print(f'has_load: {data}')
#     for sample in tqdm(cnt):
#         sample = json.loads(sample)
#         res = call_glm(prompt.replace('\[content]', sample['content'][:7420])[: 8096])
#         print(res)

#         s.append({'id': sample['id'], 'style': res})

# root_path = '/mnt/petrelfs/share_data/zhangpan/mllm/llm_data/v001-longyuan/'
# data_list = sorted(os.listdir('/mnt/petrelfs/share_data/zhangpan/mllm/llm_data/v001-longyuan'))


# save_path = '/mnt/petrelfs/weixilin/projects/MLLM/creative/classify/classify_by_glm_add_title/'
# print(save_path)
# id = 0

# import time
# for idx, data_path in tqdm(enumerate(data_list[233:],233)):
#     data_l = read_jsonl(root_path + data_path)
#     lines = []
#     for data in tqdm(data_l[:300]):
#         response = call_glm(prompt1.replace('\[content]', data['content'][:1000]+'...').replace('\[title]', data['title']).replace('\[zazhi]', data['remark']['article_from'])[: 2000])
#         lines.append({'id': data['id'], 'content': data['content'][:1000]+'...', 'title': data['title'], 'zazhi': data['remark']['article_from'],'output': response})
#         # print(f'{data["content"][:800]}\n{response}')
#         # print("-------------------------------------")
#         id += 1
#         # if id % 300 == 0:
#         #     print(lines[-1])
#     with open(save_path + f'zz{idx+1}.json', 'w', encoding='utf8') as f:
#         json.dump(lines, f, ensure_ascii=False)
# # test()

# zhihu_clean
# import sys

# root_path1 = '/mnt/petrelfs/share_data/dongxiaoyi/share_data/zhihu_clean/qwen_100k_Instruction_2_Simple.json'
# root_path2 = '/mnt/petrelfs/share_data/dongxiaoyi/share_data/zhihu_clean/qwen_77k_Instruction_2_Simple.json'
# data_list = read_json_file(root_path1) + read_json_file(root_path2)

# st = int(sys.argv[1])
# ed = int(sys.argv[2])

# save_path = f'/mnt/petrelfs/weixilin/projects/MLLM/creative/zhihu_data/zhihu{st}.json'
# print(save_path)
# lines = []
# """
# {'id': 'BkPlAv_xK0whuGQR3i7t', 'clean_content': '4月8日零时，时隔76天，武汉“解封”。有人离开，有人归来。\n在即将归来的群体中，有我们在疫情中持续关注的“武汉返乡人员”。二月初，恐惧、担忧与慌乱将我们的情绪裹挟，
# 写着，“2020年1月6日至21日武汉来青岛市人员（铁路、民航）”，一共有2257个人的信息，包括身份证号、手机号、户籍地详址、户籍地派出所、来鲁方式、车次班次等。从已有的信息判断，我也不确定是哪一方的工作人员泄露了这些信息，因
# 为表格包含了公安部门和防控部门的交叉内容。后来，有很多亲戚朋友都来询问表格的真实性，而且我还收到了近二十通骚扰电话。其中一个电话非常吓人，打通后直接问我，“你是住在xxx（我的详细住址）吗？我猜测，给我打电话的人可能在亲
# 友微信群中看到过上述表格，因为想知道自己家附近有没有从武汉回来的人，就拨打了表格中的手机号来确认。其实很多打电话来的人，根本分不清楚什么是疑似人群，什么是感染人群，以为只要从武汉回来就有问题。从那以后，我就一直担忧镇上
# 街坊的恐慌会给我的家人带来困扰。我父母在我们镇上经营着酒店，有几次服务员跟我说，听到很多街坊邻居会在背后指指点点。我觉得现在的湖北人或者在湖北的务工人员的确还是受歧视的。我的姐夫之前也是在湖北工作，前些天去襄阳出差，在
# 返回烟台机场打车时，本来怀着好意和网约车司机提前沟通，说自己是从襄阳回来的，结果话还没说完就被挂了电话。很多朋友问我，为什么不继续向泄露信息的人追究。我觉得，那些泄露信息的人也是普通人。刚开始，我也非常气愤，但现在冷
# 静了很多。虽然我明白，如果我的个人信息在网上被大范围泄露真实地发生了，我也没有任何还手之力。但是，真正让人为难的是，实际上没有任何办法去界定一个普通人的好坏。那些泄露信息的人也不是完全出于恶意，很多人知道泄露别人的隐私
# 不好，但只要周围的人都这么做了，Ta就没有了道德上的那层顾虑。但这样有好的一面，也有坏的一面的，才是一个完整的人啊，才是像我们一样有七情六欲的普通人啊。我朋友说，我的这种想法，其实是意识到“普通人的恶”，“平庸的恶”。想通这
# 点之后，我就想，人的很多本性永远不会被根除，从根源上制定法律和制度才是最有效和最直接的方法，还有要让接触一手信息的人员提高自己的职业素养，或者从技术手段上提前提防这种事。\n有了这次经历之后，我就开始很有意识地去保护自己
# 的隐私了。现在不管是去商场，还是超市都要被收集个人信息，我一般能不填真实姓名的时候就不填，能绕过去就绕过去。除了那些支付类的App，我也不会随意上传自己的身份证。以前对信息泄露能造成什么后果完全没有概念，现在我会特别在意
# , 'instruction': '请撰写一篇关于武汉疫情下，人们对于武汉返乡人员的关注和态度变化的文章。你可以根据三位受访者的故事进行展开，讲述他们在疫情下的生活以及经历的种种困扰，如个人信息被滥用、谣言纷飞等。
# 同时，讨论公众对于这些群体的态度变化以及这一过程对他们个人的影响。', 'instruction_augment': '请帮我写一篇关于武汉疫情下，人们对武汉返乡人员的态度变化的文章，第一部分描述三位受访者的生活和困难，
# 第二部分分析公众态度的变化及其影响。', 'augment_prompt': '我会给你提供一个用于引导AI生成文章的instruction，请帮把这段instruction改成大纲的形式，大纲需要符合这个格式：“请帮我写一篇关于xxx的文章，
# 第一部分为xxx，第二部分为xxx”\n文本要简短，不超过50个字。\n注意，返回的文本仅包含改写后的instruction。
# \n需要修改的instruction为：\n{}, 'title': '那些信息被泄露的武汉返乡人员，后来怎么样了？我们找他们聊了聊'}
# """
# for data in tqdm(data_list[st: ed]):
#     response = call_glm(prompt_zhihu.replace('\[title]', data['title']).replace('\[content]', data['clean_content'][:1000]).replace('\[outline]', '，'.join(data['augment_prompt'].split('，')[1:])))
#     lines.append({'id': data['id'], 'content': data['clean_content'], 'title': data['title'], 'zazhi': '知乎','output': response})
#     print(f'{data["content"][:800]}\n{response}')
    
#     with open(save_path, 'w', encoding='utf8') as f:
#         json.dump(lines, f, ensure_ascii=False)

def generate_outline():
    root_path = '/mnt/petrelfs/weixilin/projects/MLLM/creative/sft_data/tmp/prose.json'
    save_path = '/mnt/petrelfs/weixilin/projects/MLLM/creative/sft_data/tmp/prose_outline.json'
    prompt = """文章内容：\n\[content]\n\n请提炼文章主题和写作手法，用已指导创作者根据你的提炼进行写作。
    示例1：
    关于:音乐的魔力，以音乐对情感表达和心灵治愈的重要性为主题，使用感性的描写和个人体验
    示例2：
    关于:独特的美丽，以描述不同个体的独特之美和多样性为主题，使用比喻和形象描写
    示例3：
    关于:人生的追寻，以描述人生追求目标和寻找意义的旅程为主题，运用象征和对比手法
    """.strip()
    # data_path_list = sorted(os.listdir(root_path))
    dt = {}
    for idx, data in enumerate(read_json_file(root_path)):
        res = call_glm(prompt.replace('\[content]', data['content'][:8192]))
        dt[data['id']] = res
        if idx == 0: print(prompt.replace('\[content]', data['content'][:8192]), '\n', res, data['id'])
    with open(save_path, 'w', encoding='utf8') as f:
        json.dump(dt, f, ensure_ascii=False)

def generate_outline1():
    root_path = '/mnt/petrelfs/weixilin/projects/MLLM/creative/sft_data/tmp/current_poem1.json'
    save_path = '/mnt/petrelfs/weixilin/projects/MLLM/creative/sft_data/tmp/current_poem1_outline.json'
    prompt = """诗歌内容：\n\[content]\n\n请提炼诗歌主题和写作手法，用已指导创作者根据你的提炼进行创作。
    示例1：
    关于:失去与希望，以描述失去的痛苦和希望的重生为主题，运用象征和抒情的语言
    示例2：
    关于:夜空的浪漫，以描绘星空下的浪漫场景和梦幻氛围为主题，运用比喻和抒情的词句
    示例3：
    关于:忧伤的美丽，以描述忧伤中隐藏的美和情感的细腻表达为主题
    """.strip()
    # data_path_list = sorted(os.listdir(root_path))
    dt = {}
    for idx, data in enumerate(read_json_file(root_path)):
        res = call_glm(prompt.replace('\[content]', data['content'][:8192]))
        dt[data['id']] = res
        if idx == 0: print(prompt.replace('\[content]', data['content'][:8192]), '\n', res, data['id'])
    with open(save_path, 'w', encoding='utf8') as f:
        json.dump(dt, f, ensure_ascii=False)

def generate_outline2():
    root_path = '/mnt/petrelfs/weixilin/projects/MLLM/creative/sft_data/tmp/novel.json'
    save_path = '/mnt/petrelfs/weixilin/projects/MLLM/creative/sft_data/tmp/novel_outline.json'
    prompt = """小说内容：\n\[content]\n\n请根据小说内容提炼大纲，用于指导创作者根据你的提炼进行创作。提炼的大纲不超过50字。
    """.strip()
    # data_path_list = sorted(os.listdir(root_path))
    dt = {}
    for idx, data in tqdm(enumerate(read_json_file(root_path))):
        res = call_glm(prompt.replace('\[content]', data['content'][:7800]))
        dt[data['id']] = res
        if idx == 0: print(prompt.replace('\[content]', data['content'][:7800]), '\n', res, data['id'])
    with open(save_path, 'w', encoding='utf8') as f:
        json.dump(dt, f, ensure_ascii=False)
# /mnt/petrelfs/weixilin/projects/MLLM/creative/sft_data/tmp/novel.json
generate_outline2()

prompt_list = [
"""
示例1
```
🌈露腿的季节到了，作为一个资深的减肥者，今天给大家分享一个适合短期减肥的轻断食法~\n🌈🌈🌈🌈🌈🌈🌈🌈🌈🌈🌈🌈🌈🌈🌈\n1️⃣3天轻断食\n早餐:水果🍎➕鸡蛋🥚\n午餐：牛奶🥛➕鸡蛋🥚\n晚餐：酸奶🍶➕全麦吐司🥪\n这三天肚子的肉会变少很多\n腰的部位最为明显\n（体重掉的会比较多，大概2-6斤）\n2️⃣3天水果餐\n早餐：全麦吐司🥪➕鸡蛋🥚\n午餐：水果沙拉🥗\n晚餐：尽量不吃，实在太饿吃个小苹果\n（体重可以掉1-3斤）\n3️⃣3天沙拉餐\n早餐：鸡蛋🥚➕牛奶🥛\n午餐：沙拉餐（五谷，蔬菜🥬，水果沙拉🥗)\n晚餐：燕麦配酸奶🍶/牛奶🥛\n这里诚心推荐燕麦，饱腹感特别强还有清肠排毒的效果(体重大概可以瘦1-3斤)\n4️⃣：2+1巩固法\n两天沙拉餐🥗\n（三餐:五谷，蔬菜🥬水果）\n一天：正常餐\n(三餐：最好还是少吃点，鱼肉牛肉优先，连续反复吃一个星期，可以固定体重)\n✴️四个阶段下来后，千万不要去暴饮暴食\n✴️\n减肥期间一定要戒零食甜品🍰汽水油炸类等易胖的食物\n每天多喝热水有助于提高新城代谢（2000ml以上）\n前两天的轻断食有可能会轻微便秘，可以喝蜂蜜水或则吃燃脂奶片或则燕麦水来排毒\n✴️\n减肥过程中体重下降的规律\n1️⃣直线下降，每天都在降低，极个别\n2️⃣阶梯式下降，下降三五天，平衡一段时间，体重继续下降，再平衡，这是最常见的现象\n3️⃣爬山式，下降几斤，上涨一两斤，接着下降，这样的为少数\n懂了吧🌈减肥期间保持正常的心态🥰和坚持不懈都是很重要的哦\n✴️\n懒人减肥法，轻断食只适合短期减肥，长期使用营养均衡，影响健康！胃病严重，三高等人群就不适用。后续再给大家分享健康，营养均衡，包袱的营养餐🌈\n✴️\n这个夏天不减肥，防晒霜都要比别人用的多\n减肥这种事儿，不需要鸡汤\n毕竟已经到了露腿露腰的季节！\n改变自己不需要别人的鞭策\n你瘦你美，你得世界就不一样了！\n这感觉，谁瘦谁美大家都知道！
```
示例2
```
🌈夏天这么长，一定要找一个会做饮料🥤的人做朋友，🌈夏天这么热，有喝不完的饮料🍹不是在做梦喔~今天分享的几款果冻饮品，简单好喝又解暑🌞夏天必备！\n🍭1、肥宅水蜜桃冻\n🌸水蜜桃皮2个，白凉粉25克，冰糖30克，清水500克，雪碧，养乐多🌸\n①水蜜桃+盐搓洗干净\n②剥下桃皮+清水煮出粉色，捞出\n③加入冰糖和白凉粉煮融化，倒出放凉凝固\n④玻璃杯里加入蜜桃冻+雪碧+养乐多+青柠片+薄荷\n🍭 2、梦幻蝶豆花茶冻\n🌸蝶豆花5朵，清水500克，冰糖30克，白凉粉25克，柠檬水🌸\n①蝶豆花+清水煮出颜色，捞出蝶豆花扔掉\n②加入冰糖和白凉粉煮融化倒出放凉\n③玻璃杯里加入蝶豆花冻+柠檬水+柠檬片即可\n🍭 3、百香果冻柠乐\n🌸百香果5个，清水500克，冰糖30克，白凉粉25克，柠檬水🌸\n①百香果肉+清水+白糖+白凉粉煮融化\n②捞出一些百香果籽扔掉\n③玻璃杯里加入百香果冻+柠檬水+柠檬片即可；\n🍭 4、消脂茉莉花茶冻\n🌸茉莉花茶包1个，清水500克，冰糖30克，白凉粉25克，小布丁火锅救星1个，养乐多🌸\n①茉莉花茶包+清水煮出颜色，捞出茶包扔掉\n②加入冰糖和白凉粉煮融化倒出放凉\n③玻璃杯里倒入一个小布丁，加入养乐多拌匀融化，再加入茉莉花茶冻拌匀即可；\n🌈夏天又馋又想美的小伙伴们，推荐这个超级补丁的火锅救星喔[喝奶茶R]薄荷味口感清新好喝，很适合夏天🌞可冲泡在低于40度的温水或者饮料中🍹吃大餐后来一杯，保护肠胃，增加肠胃蠕动，带走肠道油脂垃圾，帮助改变体质 ，让你无论何时何地都能吃的很嗨[自拍R]
```
请参考示例风格，写一篇崇明岛旅游笔记。
""".strip(),
"""
示例1：
```
“孔乙己是站着喝酒而穿长衫的唯一的人。他身材很高大；青白脸色，皱纹间时常夹些伤痕；一部乱蓬蓬的花白的胡子。穿的虽然是长衫，可是又脏又破，似乎十多年没有补，也没有洗。他对人说话，总是满口之乎者也，叫人半懂不懂的。因为他姓孔，别人便从描红纸上的“上大人孔乙己”这半懂不懂的话里，替他取下一个绰号， 叫作孔乙己。

孔乙己一到店，所有喝酒的人便都看着他笑，有的叫道，“孔乙己，你脸上又添上新伤疤了！”他不回答，对柜里说，“温两碗酒，要一碟茴香豆。”便排出九文大 钱。他们又故意的高声嚷道，“你一定又偷了人家的东西了！”孔乙己睁大眼睛说，“你怎么这样凭空污人清白……”“什么清白？我前天亲眼见你偷了何家的书，吊着打。”孔乙己便涨红了脸，额上的青筋条条绽出，争辩道，“窃书不能算偷……窃书！……读书人的事，能算偷么？”接连便是难懂的话，什么“君子固穷”，什么“者乎”之类，引得众人都哄笑起来：店内外充满了快活的空气。”
```
示例2：
```
“闲人还不完，只撩他，于是终而至于打。阿Q在形式上打败了，被人揪住黄辫子，在壁上碰了四五个响头，闲人这才心满意足的得胜的走了，阿Q站了一刻，心里想，“我总算被儿子打了，现在的世界真不像样……”于是也心满意足的得胜的走了。
阿Q想在心里的，后来每每说出口来，所以凡是和阿Q玩笑的人们，几乎全知道他有这一种精神上的胜利法，此后每逢揪住他黄辫子的时候，人就先一着对他说：
“阿Q，这不是儿子打老子，是人打畜生。自己说：人打畜生！”
阿Q两只手都捏住了自己的辫根，歪着头，说道：
“打虫豸，好不好？我是虫豸——还不放么？”
但 虽然是虫豸，闲人也并不放，仍旧在就近什么地方给他碰了五六个响头，这才心满意足的得胜的走了，他以为阿Q这回可遭了瘟。然而不到十秒钟，阿Q也心满意足 的得胜的走了，他觉得他是第一个能够自轻自贱的人，除了“自轻自贱”不算外，余下的就是“第一个”。状元不也是“第一个”么？“你算是什么东西”呢！？
阿Q以如是等等妙法克服怨敌之后，便愉快的跑到酒店里喝几碗酒，又和别人调笑一通，口角一通，又得了胜，愉快的回到土谷祠，放倒头睡着了。
```
示例3：
```
“长妈妈，已经说过，是一个一向带领着我的女工，说得阔气一点，就是我的保姆。我的母亲和许多别的人都这样称呼她，似乎略带些客气的意思。只有祖母叫她阿长。 我平时叫她“阿妈”，连“长”字也不带；但到憎恶她的时候，——例如知道了谋死我那隐鼠的却是她的时候，就叫她阿长。
……
虽然背地里说人长短不是好事情，但倘使要我说句真心话，我可只得说：我实在不大佩服她。最讨厌的是常喜欢切切察察，向人们低声絮说些什么事。还竖起第二个手 指，在空中上下摇动，或者点着对手或自己的鼻尖。我的家里一有些小风波，不知怎的我总疑心和这“切切察察”有些关系。又不许我走动，拔一株草，翻一块石 头，就说我顽皮，要告诉我的母亲去了。一到夏天，睡觉时她又伸开两脚两手，在床中间摆成一个“大”字，挤得我没有余地翻身，久睡在一角的席子上，又已经烤得那么热。推她呢，不动；叫她呢，也不闻。”
```
请仿照给定示例的文风，写一篇文章，主题是与冬天有关。
""".strip(),
    "请以金庸武侠小说的风格，写一个白雪公主和农夫的故事。"
]

# test_cases = [
#     '一个人乐意去探索陌生世界，仅仅是因为好奇心吗？请写一篇高考作文，谈谈你对这个问题的认识和思考。要求：(1） 自拟题目；（2）不少于 800字。',
#     '吹灭别人的灯，并不会让自己更加光明；阻挡别人的路，也不会让自己行得更远。“一花独放不是春，百花齐放春满园。”如果世界上只有一种花朵，就算这种花朵再美，那也是单调的。以上两则材料出自习近平总书记的讲话，以生动形象的语言说出了普遍的道理。请据此写一篇高考作文，体现你的认识与思考。要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不得泄露个人信息；不少于800字。',
#     '中国共产主义青年团成立100周年之际，中央广播电视总台推出微纪录片，介绍一组在不同行业奋发有为的人物。他们选择了自己热爱的行业，也选择了事业创新发展的方向，展示出开启未来的力量。有位科学家强调，实现北斗导航系统服务于各行各业，“需要新方法、新思维、新知识”。她致力于科技攻关，还从事科普教育，培育青少年的科学素养。有位摄影家认为，“真正属于我们的东西，是民族的，血脉的，永不过时”。他选择了从民族传统中汲取养分，通过照片增强年轻人对中国文化的认同。有位建筑家主张，要改变“千城一面”的模式，必须赋予建筑以理想和精神。他一直努力建造“再过几代人仍然感觉美好”的建筑作品。复兴中学团委将组织以“选择·创造·未来”为主题的征文活动，请结合以上材料写一篇高考作文，体现你的认识与思考。要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不得泄露个人信息；不少于800字。',
#     '我们公司最近新生成了一款电动牙刷，主打质量高，续航长，请你写一篇双11的运营稿，要求广告味不要那么浓，尽可能让用户愿意阅读和关注，愿意前往我们的京东官方店铺查看产品详情，大约500字',
#     'OpenAI近期爆发了董事会冲突，两位重要创始人被迫离职，请你写一篇大模型运营稿，结合热点新闻，与我们自己在研发的大模型建立联系，目标是希望这个运营稿能帮助吸引更多微信用户阅读，大约1000字。',
#      "上海人工智能实验室开发了一款支持千亿参数的大语言模型，请以小红书的风格介绍一下吧",
#      "请根据我给出的[外星人入侵、核弹、流亡]这些关键词来撰写一篇[科幻]题材的短篇故事。 \n故事需要拥有[引人入胜]的开头以及[反转]的结局，故事线[跌宕起伏]。\n注意请使用[刘慈欣]的写作风格为我撰写这篇故事。减少赘述，内容中不要有重复或意思相近的段落，大约800字",
#      "有个梗是这么说的，写word的工资没写Excel的工资高，而做Excel的工资又不如写PPT的工资高，当然写PPT的不如讲PPT的，讲PPT的不如听PPT的。请根据这个梗来写一个笑话，大约500字",
#      "\"凌晨12点，小明还没有困意，在窗边思考着某段代码怎么写\"，请接着往下，延续一篇 300 字的恐怖故事",
#      "以《功守道》为题，讲一个关于达摩院的武侠故事，大约500字",
#      "请写不少于1000字的推理小说，希望有对话环节，请用中文，信息如下:1.这是一起发生在校园的盗窃案\n2主人公是一位私人侦探\n3.故事中需要有一个谜团，让读者不断导找线索，直到最后真相大白4.故事中需委引入若干名嫌疑犯，让读者不断猜测四手5.为了帮助读者推理，需要在故事中提供一些有用的线索6.故事的最终结局应该把所有疑团都解开，并告诉读者凶手是谁7.最重要的是，要有剧情反转，惊艳到读者",
#     "请仿照史记的风格，写一段文言文描述下文所介绍的事件，大约500字：\n大业十一年，云定兴被授以左屯卫大将军，奉命援救在雁门关被突厥始毕可汗所率大军围困的隋炀帝。隋炀帝派人把诏书绑在木头上，放进汾河让诏书顺流而下，希望有人看到诏书前来救援。云定兴向各地招募愿意出征的军士，李世民那年只有十六岁，前去应募从军，被划归云定兴的帐下。云定兴手下只有两万新兵，且多是步兵。李世民向云定兴建议：突厥敢围困天子，是认定我们没有援军。不如我们把军队前后拉开，延绵数十里，让敌军白天看见旌旗招展，晚上听见钲鼓声声，误以为大军压境，如此才能不战而胜。若他们知我虚实，两兵相接，则胜败难料。云定兴采纳了李世民的疑兵之计攻突厥，突厥兵看到隋军浩浩荡荡络绎不绝，果然以为隋军大批救兵到，于是解围撤退。",
#     "请写一篇劝学的文言文，大约500字，文章以一位朝中官员的口吻，叙述个人早年虚心求教和勤苦学习的经历，生动而具体地描述了自己借书求师之难，饥寒奔走之苦，并与太学生优越的条件加以对比，有力地说明学业能否有所成就，主要在于主观努力，不在天资的高下和条件的优劣，以勉励青年人珍惜良好的读书环境，专心治学。",
#     "作一首离别的宋词，词牌名雨霖铃",
#     "请写一首关于梅花的七言绝句唐诗",
#     "请写一首关于白雪公主的伤感的现代诗",
#     "请写一首五言绝句来赞美新中国成立的意义。",
#     "创作一首以“我喜欢你”为首的藏头诗",
#     "请写一篇《寻梦环游记》的豆瓣影评，探讨电影中的家庭情感和梦想追求，大约500字。",
#     "长乐路上新开了一家清吧，老板说帮忙在大众点评上写好评会送我一杯酒，请帮我写一篇不敷衍的评价，突出环境优美，大约200字",
#     "老板忘记在小明的外卖里加番茄酱了，麻烦安抚一下他不开心 的感觉",
#     "你是一个童话大王，你现在把童话故事《白雪公主》重新改写结局，要求白雪公主嫁给了一个农夫，不少于500字。",
#     "请用金庸的风格重新改写一下童话故事《白雪公主》，不少于500字",
#     "请写一个跟动物有关的笑话",
#     "给我写一个关于减肥的笑话",
#     "你是一个悲剧作家，请改写《青蛙王子》的结局，要求青蛙王子始终是青蛙，并没有变成王子和公主在一起，不少于500字。",
#     "请写一个去北极吃火锅的旅行攻略",
#     "写一篇关于探险的学术论文，风格是幽默的，字数大约800字左右",
#     "请以《如何吃饭？》为题，写一个生活经验，字数400字左右",
#     '请以《赶路的旅人》写一篇高考作文，字数不少于800字',
#     '请写一首赞美白雪公主的现代诗歌。',
#     '请写一篇悲伤的散文，主题是猫和老鼠，字数大约600字。',
#     '请以“基于大模型的风格写作”为题写一篇学术论文',
#     '请以“基于大模型的风格写作”为题写一篇论文',
#     '你是一个小红书博主，请以“白雪公主和农夫”为主题写一篇100字左右的小红书',
#     '请以“白雪公主和农夫”为主题写一篇100字左右的学术论文',
#     '请写一篇小红书博文，主题是“北海道旅行笔记”',
# ]

# wenyanwen_test_cases = [
#         '请写一篇关于怀才不遇的文言文，讲述我得不到领导赏识，多年未级别调动的感慨，大约300字',
#         '想象并描述一个温暖的家庭生活场景，生成一篇关于亲情主题的文言文，大约300字',
#         '请仿照史记的风格，写一段文言文描述下文所介绍的事件，大约500字大业十一年，云定兴被授以左屯卫大将军，奉命援救在雁门关被突厥始毕可汗所率大军围困的隋炀帝。隋炀帝派人把诏书绑在木头上，放进汾河让诏书顺流而下，希望有人看到诏书前来救援。云定兴向各地招募愿意出征的军士，李世民那年只有十六岁前去应募从军，被划归云定兴的帐下。云定兴手下只有两万新兵，且多是步兵。李世民向云定兴建议:突厥敢围困天子，是认定我们没有援军。不如我们把军队前后拉开，延绵数十里，让敌军白天看见旌旗招展，晚上听见证鼓声声，误以为大军压境如此才能不战而胜。若他们知我虚实，两兵相接，则胜败难料。云定兴采纳了李世民的疑兵之计攻突厥，突厥兵看到隋军浩浩荡荡络绎不绝，果然以为隋军大批救兵到于是解围撤退。',
#         '请仿照史记的风格，写一段文言文描述下文所介绍的事件，大约500字:中平六年，大将军何进欲除宦官，为加强自己力量，何进下令调动董卓部进洛阳。时董卓屯河东郡，进军至绳池时朝廷派遣种劲宣读归还敕令，董卓狐疑朝廷有变，但在种劲的坚持下退至夕阳亭。 结果董卓部汉军进洛阳前，何进已经被宦官处死。公卿以下与董卓共迎汉帝于邱山阪下。初入雏阳时兵力只有3千人，为求营造大军压境的场面以震慑邻近诸侯，每晚令士兵出城，翠日再大张旗鼓入城，令到雏阳全城有大军源源不绝进军之虚况。不久令其弟董是联合吴匡杀掉上司何苗，又招揽吕布杀掉丁原，很快就吞并附近两大军阀兵力。随后董卓废少帝，立刘协即位，且不久就弑害少帝及何太后，专断朝政。',
#         '请写一篇劝学的文言文，大约500字，文章以一位朝中官员的口吻，叙述个人早年虚心求教和勤苦学习的经历，生动而具体地描述了自己借书求师之难，饥寒奔走之苦并与太学生优越的条件加以对比，有力地说明学业能否有所成就，主要在于主观努力，不在天资的高下和条件的优劣，以勉励青年人珍惜良好的读书环境，专心治学。',
#         '请写一篇文言文，以被扣留在匈奴的苏武的口吻，抒发自己的志向',
#         '请写一篇感叹赋税得役沉重的文言文',
#         '请写一篇记录钱塘江观潮的文言文',
#         '请写一篇记录登泰山的文言文，文中需有景色描写和怀古',
#         '请写一篇文言文以论述以下题目:周唐外重内轻，秦魏外轻内重各有得论',
#         '请写一篇文言文以解释这句话的含义:大学之道，在明明德，在亲民，在止于至善',
#         '请仿照聊斋志异的风格，写一篇发生在湖边的志怪小说',
#     ]
# save_path = '/mnt/petrelfs/weixilin/projects/MLLM/creative/test/chatglm3_wenyanwen.json'
# print(save_path)
# lines = []
# for test_case in wenyanwen_test_cases:
#         # test_case = tag(test_case)
#     res = call_glm(test_case)
#     lines.append({'content': test_case, 'output': res})
# with open(save_path, 'w', encoding='utf8') as f:
#     json.dump(lines, f, ensure_ascii=False)

