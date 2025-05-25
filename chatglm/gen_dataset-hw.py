# week 5 作业-2 
# 可选择作业

# 构造微调训练数据集

# - 使用 LangChain 构造训练数据样例
#     - 基于 ChatGPT 设计 `System Role` 提示词
#     - 使用 `OpenAI GPT-4o-mini` 生成基础数据
#     - 解析 OpenAI GPT 生成的训练数据
#     - 持久化存储`dataset.csv`训练数据集文件
#     - 使用 ChatGPT 实现训练数据多样化
# - 自动化批量生成训练数据集
#     - 整理收集原始数据`raw_data.txt`
#     - 自动解析原始数据样例 `raw_data_content[]`
#     - 设计 `gen_data` 训练数据生成器函数
#     - 设计训练数据生成流水线



## 使用 OpenAI SDk 构造训练数据
#
#

from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {
      "role": "system",
      "content": "你是中国古典哲学大师，尤其擅长周易的哲学解读。\n\n接下来，你收到的都是关于周易卦象的解释，你需要整理润色，并生成用于大模型训练的内容和格式。\n\n示例输入：\n\n师卦，此卦是异卦相叠，下卦为坎，上卦为坤。“师”指军队。坎为水、为险；坤为地、为顺，喻寓兵于农。兵凶战危，用兵乃圣人不得已而为之，但它可以顺利无阻碍地解决矛盾，因为顺乎形势，师出有名，故能化凶为吉。占得此卦，对于军事上率师出征非常有利，必无灾祸。师卦是天马出群之卦，以寡伏众之象。\n师卦位于讼卦之后，《序卦》之中这样解释道：“讼必有众起，故受之以师。师者，众也。”争讼的人越来越多，以致形成了军队。\n\n期待结果：\n\ncontent:\"师卦\"\nsummary:\"在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表“师”，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，暗示着通过将军事力量安置于民间，可以在必要时顺利调动。\n\n师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。\n\n师卦紧随讼卦（争讼卦），在《序卦》中解释为“讼必有众起，故受之以师”。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。\""
    },
    {
      "role": "user",
      "content": "蒙卦是教育启蒙的智慧，艮为山，坎为泉，山下出泉。泉水始流出山，则必将渐汇成江河,正如蒙稚渐启，又山下有险，因为有险停止不前，所以蒙昧不明。事物发展的初期阶段，必然蒙昧，所以教育是当务之急，养学生纯正无邪的品质，是治蒙之道。\n蒙卦，这个卦是异卦相叠，下卦为坎，上卦为艮。艮是山的形象，喻止；坎是水的形象，喻险。卦形为山下有险，仍不停止前进，是为蒙昧，故称蒙卦。但因把握时机，行动切合时宜;因此，具有启蒙和通达的卦象。\n《蒙》卦是《屯》卦这个始生卦之后的第二卦。《序卦》中说：“物生必蒙，故受之以蒙。蒙者，蒙也，特之稚也。”物之幼稚阶段，有如蒙昧未开的状态，在人则是指童蒙。\n《象》中这样解释蒙卦：山下出泉，蒙；君子以果行育德。"
    },
    {
      "role": "assistant",
      "content": "content:\"蒙卦\"\nsummary:\"在周易中，蒙卦象征着教育启蒙和智慧的初期阶段，由两个异卦相叠组成：下卦坎（水）和上卦艮（山）。坎象征水，艮象征山，代表了山下有水源，水流出山。这预示着在事物发展的起初阶段，人们会陷入蒙昧状态，因此教育成为当务之急。只有培养纯正无邪的品质，才能走上通达之路，如同泉水汇成江河。\n\n蒙卦的核心哲学是：事物发展的初期阶段必然处于蒙昧状态，但如果能把握时机，行动切合时宜，就能启蒙通达。蒙卦紧随始生卦《屯》之后，《序卦》中解释道：“物生必蒙，故受之以蒙。蒙者，蒙也，特之稚也。”这说明物在幼稚阶段都会经历蒙昧状态，找它蒙即是稚嫩。\n\n《象》中解释说：山下出泉，蒙；君子以果行育德，意味着像泉水从山下流出一样，启蒙教育有助于培育美德。\""
    },
    {
      "role": "user",
      "content": "需卦，本卦是异卦相叠，上卦为坎，下卦为乾。乾卦象征天空辽阔，坎卦象征乌云密布。雨水之于传统农业社会而言，是最大的需求，因此用这个意象来表现。古人引申为雨将下，万物皆待，所以叫需。\n需，古体字结构是“雨上天下”，从雨而声。如苍天下雨，滋养万物及人。从人的角度看，天下雨，则不易出行耕作，因此在家喝酒吃肉、静待天时。所以，这一卦的基本象征就是等待、不进，还有饮食之义。\n \n需卦位于蒙卦之后，《序卦》之中这样解释道：“物稚不可不养也，故受之以需。需者，饮食之道也。”有所需要，也是有所等待。\n《象》曰：云上于天，需;君子以饮食宴乐。\n《象》中这段话的意思是说：需卦的卦象是乾(天)下坎(水)上，为水在天上之表象。水汽聚集天上成为云层，密云満天，但还没有下雨，需要等待;君子在这个时候需要吃喝，饮酒作乐，即在等待的时候积蓄力量。\n需卦象征等待，启示人们守正待机的道理，属于中上卦。《象》这样评断此卦：明珠土埋日久深，无光无亮到如今，忽然大风吹土去，自然显露有重新。\n"
    },
    {
      "role": "assistant",
      "content": "content:\"需卦\"\nsummary:\"在周易中，需卦是一个富有深意的卦象，由两个异卦相叠组成：下卦乾（天）和上卦坎（水）。乾象征广阔的天空，而坎象征乌云密布，暗示着迫切需要雨水的意象。古人将雨水视为万物生长的必需，因此命名为“需”。需卦的基本象征是等待和饮食之道。\n\n需卦位于蒙卦之后，《序卦》中解释为“物稚不可不养也，故受之以需。需者，饮食之道也。”说明物在幼稚状态需要得到滋养，因此受到需卦的启示。《象》中描述：云上于天，需；君子以饮食宴乐。意味着天空乌云密布，需要等待雨水，而君子则在等待的过程中积蓄力量，享受饮食和欢乐。\n\n需卦的核心哲学是：等待，启示着守正待机的道理。它属于中上卦，象征着埋藏已久的明珠，经过大风吹去尘埃后，重新显露光芒。\""
    }
  ],
  temperature=1,
  max_tokens=4095,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)




## 使用 LangChain 构造训练数据
#
#


from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

chat = ChatOpenAI(model="gpt-4o-mini",
                  temperature=1,
                  max_tokens=4095)

system_content = """
你是中国古典哲学大师，尤其擅长周易的哲学解读。

接下来，你收到的都是关于周易卦象的解释，你需要整理润色，并生成用于大模型训练的内容和格式。

示例输入：

师卦，此卦是异卦相叠，下卦为坎，上卦为坤。“师”指军队。坎为水、为险；坤为地、为顺，喻寓兵于农。兵凶战危，用兵乃圣人不得已而为之，但它可以顺利无阻碍地解决矛盾，因为顺乎形势，师出有名，故能化凶为吉。占得此卦，对于军事上率师出征非常有利，必无灾祸。师卦是天马出群之卦，以寡伏众之象。
师卦位于讼卦之后，《序卦》之中这样解释道：“讼必有众起，故受之以师。师者，众也。”争讼的人越来越多，以致形成了军队。

期待结果：

content:"师卦"
summary:"在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表“师”，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，暗示着通过将军事力量安置于民间，可以在必要时顺利调动。

师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。

师卦紧随讼卦（争讼卦），在《序卦》中解释为“讼必有众起，故受之以师”。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。"
"""

# 原始数据
raw_content = (
    "蒙卦是教育启蒙的智慧，艮为山，坎为泉，山下出泉。泉水始流出山，则必将渐汇成江河,正如蒙稚渐启，又山下有险，因为有险停止不前，所以蒙昧不明。事物发展的初期阶段，"
    "必然蒙昧，所以教育是当务之急，养学生纯正无邪的品质，是治蒙之道。\n蒙卦，这个卦是异卦相叠，下卦为坎，上卦为艮。艮是山的形象，喻止；坎是水的形象，喻险。卦形为山下有险，仍不停止前进，"
    "是为蒙昧，故称蒙卦。但因把握时机，行动切合时宜;因此，具有启蒙和通达的卦象。\n《蒙》卦是《屯》卦这个始生卦之后的第二卦。《序卦》中说：“物生必蒙，故受之以蒙。蒙者，蒙也，特之稚也。”物之幼稚阶段，"
    "有如蒙昧未开的状态，在人则是指童蒙。\n《象》中这样解释蒙卦：山下出泉，蒙；君子以果行育德。"
    )

messages = [
    SystemMessage(
        content=system_content
    ),
    HumanMessage(
        content=raw_content
    ),
]

ai_message = chat(messages)




## 解析 OpenAI GPT 生成的训练数据
#
#

ai_message.content

text = ai_message.content

# 分割字符串来找到content和summary的位置
content_start = text.find('content:"') + len('content:"')
content_end = text.find('"\nsummary:')
summary_start = text.find('summary:"') + len('summary:"')
summary_end = text.rfind('"')

# 提取并存储content和summary
content = text[content_start:content_end].strip()
summary = text[summary_start:summary_end].strip()

print("Content:", content)
print("Summary:", summary)




### 持久化存储训练数据集文件
#
#

import csv

# 如果没有GPT API，可以使用预定义的变量
# content = "蒙卦"
# summary = "在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表“师”，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，
#            暗示着通过将军事力量安置于民间，可以在必要时顺利调动。师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。
#            因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。师卦紧随讼卦（争讼卦），在《序卦》中解释为“讼必有众起，故受之以师”。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。"

# 新建CSV文件并写入数据
with open('test_dataset.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入标题行
    writer.writerow(['content', 'summary'])
    # 写入数据行
    writer.writerow([content, summary])




### 数据增强：构造多样化的提问方式
#
#


def generate_question_summary_pairs(content, summary):
    """
    生成20对提问和总结的配对。

    :param content: 内容（例如：“蒙卦”）。
    :param summary: 内容的总结。
    :return: 包含20对提问和总结的列表。
    """
    # 20种提问模板
    question_templates = [
        "{}代表什么？",
        "周易中的{}含义是什么？",
        "请解释一下{}。",
        "{}在周易中是什么象征？",
        "周易{}的深层含义是什么？",
        "{}和教育启蒙有什么联系？",
        "周易的{}讲述了什么？",
        "{}是怎样的一个卦象？",
        "{}在周易中怎样表达教育的概念？",
        "{}的基本意义是什么？",
        "周易中{}的解释是什么？",
        "{}在周易中代表了哪些方面？",
        "{}涉及哪些哲学思想？",
        "周易中{}的象征意义是什么？",
        "{}的主要讲述内容是什么？",
        "周易{}的核心思想是什么？",
        "{}和启蒙教育之间有何联系？",
        "在周易中，{}象征着什么？",
        "请描述{}的含义。",
        "{}在周易哲学中扮演什么角色？"
    ]

    # 使用content填充提问模板
    questions = [template.format(content) for template in question_templates]

    # 创建提问和总结的配对
    question_summary_pairs = [(question, summary) for question in questions]

    return question_summary_pairs

import csv

# 如果没有GPT API，可以使用预定义的变量
# content = "蒙卦"
# summary = "在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表“师”，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，
#            暗示着通过将军事力量安置于民间，可以在必要时顺利调动。师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。
#            因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。师卦紧随讼卦（争讼卦），在《序卦》中解释为“讼必有众起，故受之以师”。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。"

pairs = generate_question_summary_pairs(content, summary)

# 将结果写入CSV文件
with open('test_dataset.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['content', 'summary'])
    for pair in pairs:
        writer.writerow(pair)


## 自动化批量生成训练数据流水线
#
#
#

# 初始化一个空列表用于存储原始内容数据
raw_content_data = []

# 读取文件并分割数据样例
with open('data/raw_data.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    # 使用连续的换行符('\n\n')作为分隔符来分割文本
    data_samples = content.split('\n\n')

    # 遍历分割后的数据样例并添加到列表中
    for sample in data_samples:
        # 移除每个样例中的额外空白字符（如果有的话）
        cleaned_sample = sample.strip()
        # 仅添加非空样例
        if cleaned_sample:
            raw_content_data.append(cleaned_sample)


# 输出结果以验证
for i, sample in enumerate(raw_content_data[:5]):  # 打印前5个样例以检查
    print(f"样例 {i+1}:")
    print(sample)
    print("------")




from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# 初始化LangChain的GPT-4o-mini调用
chat = ChatOpenAI(model="gpt-4o-mini",
                  temperature=1,
                  max_tokens=4095)

def gen_data(raw_content):
    """
    使用LangChain GPT-4o-mini调用处理单个数据样例。

    :param raw_content: 原始数据样例。
    :return: GPT-4o-mini模型生成的内容。
    """
    # 系统消息定义背景和任务
    system_message = SystemMessage(
        content="""
        你是中国古典哲学大师，尤其擅长周易的哲学解读。

        接下来，你收到的都是关于周易卦象的解释，你需要整理润色，并生成用于大模型训练的内容和格式。

        示例输入：

        师卦，此卦是异卦相叠，下卦为坎，上卦为坤。“师”指军队。坎为水、为险；坤为地、为顺，喻寓兵于农。兵凶战危，用兵乃圣人不得已而为之，但它可以顺利无阻碍地解决矛盾，因为顺乎形势，师出有名，故能化凶为吉。占得此卦，对于军事上率师出征非常有利，必无灾祸。师卦是天马出群之卦，以寡伏众之象。
        师卦位于讼卦之后，《序卦》之中这样解释道：“讼必有众起，故受之以师。师者，众也。”争讼的人越来越多，以致形成了军队。

        期待结果：

        content:"师卦"
        summary:"在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表“师”，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，暗示着通过将军事力量安置于民间，可以在必要时顺利调动。

        师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。

        师卦紧随讼卦（争讼卦），在《序卦》中解释为“讼必有众起，故受之以师”。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。"
        """
    )

    # 人类消息包含原始数据样例
    human_message = HumanMessage(
        content=raw_content
    )

    # 构建消息列表并进行模型调用
    messages = [system_message, human_message]
    ai_message = chat(messages)

    return ai_message.content


# 示例调用（使用 raw_data.txt 中解析的数据样例）
generated_content = gen_data(raw_content_data[0])
print(generated_content)


def dataset_parser(ai_message_content):
    """
    解析由gen_data函数生成的ai_message.content，提取content和summary。

    :param ai_message_content: gen_data函数返回的文本。
    :return: 提取的content和summary。
    """
    # 分割字符串来找到content和summary的位置
    content_start = ai_message_content.find('content:"') + len('content:"')
    content_end = ai_message_content.find('"\nsummary:')
    summary_start = ai_message_content.find('summary:"') + len('summary:"')
    summary_end = ai_message_content.rfind('"')

    # 提取并存储content和summary
    content = ai_message_content[content_start:content_end].strip()
    summary = ai_message_content[summary_start:summary_end].strip()

    return content, summary


# 示例调用（使用假设的gen_data函数返回的文本）
content, summary = dataset_parser(generated_content)
print("Content:", content)
print("Summary:", summary)

import csv
import datetime
import os

def main():
    # 确保 data 目录存在
    if not os.path.exists('data'):
        os.makedirs('data')

    # 解析 data/raw_data.txt 得到 raw_content_data 列表
    raw_content_data = []
    with open('data/raw_data.txt', 'r', encoding='utf-8') as file:
        content = file.read()
        data_samples = content.split('\n\n')
        for sample in data_samples:
            cleaned_sample = sample.strip()
            if cleaned_sample:
                raw_content_data.append(cleaned_sample)

    # 创建带有时间戳的CSV文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/zhouyi_dataset_{timestamp}.csv"

    # 创建CSV文件并写入标题行
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['content', 'summary'])

        # 循环遍历 raw_content_data 数据样例
        for raw_content in raw_content_data:
            # 调用 gen_data 方法得到 ai_message_content
            ai_message_content = gen_data(raw_content)

            # 解析 ai_message_content 得到 content 和 summary
            content, summary = dataset_parser(ai_message_content)
            
            print("Content:", content)
            print("Summary:", summary)

            # 调用 generate_question_summary_pairs 得到20组 pairs
            pairs = generate_question_summary_pairs(content, summary)

            # 将 pairs 写入 csv 文件
            for pair in pairs:
                writer.writerow(pair)

# 执行主函数
main()


def gen_data(raw_content):
    """
    使用LangChain GPT-4o-mini调用处理单个数据样例。

    :param raw_content: 原始数据样例。
    :return: GPT-4o-mini模型生成的内容。
    """
    # 系统消息定义背景和任务
    system_message = SystemMessage(
        content="""
        你是中国古典哲学大师，尤其擅长周易的哲学解读。

        接下来，你收到的都是关于周易卦象的解释，你需要整理润色，并生成用于大模型训练的内容和格式。

        示例输入：

        师卦，此卦是异卦相叠，下卦为坎，上卦为坤。“师”指军队。坎为水、为险；坤为地、为顺，喻寓兵于农。兵凶战危，用兵乃圣人不得已而为之，但它可以顺利无阻碍地解决矛盾，因为顺乎形势，师出有名，故能化凶为吉。占得此卦，对于军事上率师出征非常有利，必无灾祸。师卦是天马出群之卦，以寡伏众之象。
        师卦位于讼卦之后，《序卦》之中这样解释道：“讼必有众起，故受之以师。师者，众也。”争讼的人越来越多，以致形成了军队。

        期待结果：

        content:"师卦"
        summary:"在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表“师”，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，暗示着通过将军事力量安置于民间，可以在必要时顺利调动。

        师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。

        师卦紧随讼卦（争讼卦），在《序卦》中解释为“讼必有众起，故受之以师”。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。"

        返回格式要求：
        content:"{卦名}"
        summary:"{内容}"
        """
    )

    # 人类消息包含原始数据样例
    human_message = HumanMessage(
        content=raw_content
    )

    # 构建消息列表并进行模型调用
    messages = [system_message, human_message]
    ai_message = chat(messages)

    return ai_message.content

# 执行主函数
main()