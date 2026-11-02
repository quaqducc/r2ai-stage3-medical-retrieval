"""zh_kb host rules on real fixture pages: rule used, a known content sentence kept, known chrome dropped."""
import pytest

from conftest import fixture_rows
from extract import parsers
from extract.clean import clean_markdown

# (host, page id, sentences that must be present, strings that must be absent)
CASES = [
    ("zysjonline.com", 4256781, ["产后寒中感寒，及饮食冷物，下攻小腹作痛"], ["本篇方剂", "宗金鉴》卷四十八", "方向键翻页"]),
    ("zysjonline.com", 4286796, ["因心肺二经．被瘀血所侵．孔窍皆闭．"], ["方向键翻页", "选择格式", "验证码"]),
    ("zhongyibaodian.net", 4090227, ["＊改善视网膜循环障碍（糖尿病、高血压、动脉硬化等）", "药物可增加静脉张力及起到保护血管的作用。"],
     ["个分类", "中医宝典 >", "跳转到"]),
    ("zhongyibaodian.net", 4110144, ["由环氧氯丙烷于90～95°C、硫酸溶液中水解再经分馏制得。"],
     ["塑胶添加剂", "个分类", "参考资料", "Record of CAS RN", "外部链接"]),
    ("www.a-hospital.com", 3000049, ["乳房痈疽缠绵不愈而成痨者"], ["转载请保留此链接", "目前暂无留言", "A+医学百科 >>", "参看"]),
    ("www.a-hospital.com", 3091273, ["玫瑰疹（rose spots）是一种鲜红色的圆形斑疹"],
     ["按症状名称开头字母", "点击部位名称", "症状查询专题", "转载请保留此链接"]),
    ("baby.39.net", 791235, ["宝宝1岁零5个月，非常认人，或者说是怕生"], ["中国领先的健康门户网站", "责任编辑", "专稿", "栏目关注"]),
    ("baby.39.net", 767163, ["该期主要适用于乳牙反"], ["栏目关注", "声明：39健康网", "擅长领域"]),
    ("byby.39.net", 794252, ["一位中年男子就诊，说自己的性功能有毛病了"], ["我要评论", "实习编辑", "以上内容仅授权"]),  # 2010 layout
    ("shen.39.net", 903404, ["透析就是最好的例证"], ["相关阅读", "尿毒症治疗偏方可信吗？"]),
    ("gc.39.net", 853200, ["对于年幼患儿应该予以恰当的喂养"], ["推荐阅读", "【治便秘总依赖药物不可行】"]),
    ("fitness.39.net", 817159, [], ["39减肥健身学院", "想要健康瘦下来吗"]),
    ("jbk.39.net", 2528104, ["治疗费用：不同医院收费标准不一致，市三甲医院约（500-1000元）"],
     ["其他类似疾病", "擅长：", "用药经验", "详细"]),
    ("jbk.39.net", 2520688, ["尿液又称理学或肉眼检查，包括尿量、颜色、透明度、气味和比密（比重）的检查。"],
     ["地图样骨缺损", "[]", "查看详细", "- 科室："]),
    ("test.pmphai.com", 2545751, ["毛息肉（hairy polyp）是一种少见的口咽部或鼻咽部的先天性畸形"],
     ["此内容为收费内容", "临床表现", "京ICP备"]),
    ("test.pmphai.com", 2540228, ["右眼脉络膜转移癌；右眼视网膜脱离；右眼继发性青光眼；双眼老年性白内障"],
     ["上一篇", "此内容为收费内容", "收藏"]),
    ("www.msdmanuals.cn", 3980738, ["医生根据患者的症状和皮疹外观以及所在身体部位来诊断皮炎。"],
     ["Test your Knowledge", "Merck &", "完整评审"]),
    ("www.msdmanuals.cn", 3982327, ["产后出血是指分娩后24小时内失血量>1000毫升"], ["Test your Knowledge", "同行评审人"]),
    ("www.wujue.com", 2011388, ["惊震内障是指头眼部挫伤或眼部锐器伤"], ["疾病录入", "网友评论", "发表评论"]),
    ("www.wujue.com", 2006529, ["脱疽是指四肢末端坏死"], ["下一页", "上一篇文章"]),
    ("jb39.com", 2516109, ["遗传性运动失调性多发性神经炎又称Refsum病"], ["《家庭医学百科-医疗康复篇》", "显示全部", "上一页"]),
    ("jb39.com", 2514696, ["10岁以下多见，偶见于成人。"], ["肝囊肿", "囊肿 - 骨", "显示全部"]),
    ("www.zydcd.com", 2052458, ["三年前左下颌开始酸痛，以后坚硬如石"], ["内容推荐", "关于本站", "Copyright"]),
    ("www.zydcd.com", 2113838, ["病证名。见《证治准绳·杂病》。"], ["内容推荐", "关于本站", "首页"]),
    ("health.people.com.cn", 856591, ["猪肝维生素A含量为4972微克RAE/天"], ["人民网服务邮箱", "热词", "点击播报本文"]),
    ("health.people.com.cn", 857273, ["仅凭面部特征判断健康情况不科学也不准确。"], ["分享让更多人看到", "已订阅", "点击播报本文"]),
    ("qihuangzhishu.com", 899675, ["狼疮带试验是检查系统性红斑狼疮（SLE）病人"], ["电子书打不开", "方剂主治", "下一页："]),
    ("qihuangzhishu.com", 886851, ["【菊花】 （味甘苦性凉，去梗净晒干。）清心明目，疏热散风。"], ["1000余本中医古籍", "下载所有中医书籍"]),
]

# Remaining *.39.net article channels share one rule; one fixture page each, generic checks only.
ART39 = ["care.39.net", "heart.39.net", "fk.39.net", "nk.39.net", "cancer.39.net", "tnb.39.net", "woman.39.net",
         "food.39.net", "ek.39.net", "man.39.net", "gan.39.net", "wei.39.net", "sj.39.net", "gk.39.net", "pf.39.net"]
ART39_JUNK = ["中国领先的健康门户网站", "擅长领域", "实习编辑", "专稿，未经书面授权", "以上内容仅授权39健康网", "举报/反馈"]


def _parse(host, pid=None):
    rows = [r for r in fixture_rows(host) if pid is None or r[0] == pid]
    assert rows, f"fixture page {host} {pid} missing"
    out = []
    for _, url, html in rows:
        doc = parsers.parse(host, url, html)
        out.append((doc, clean_markdown(doc.markdown, doc.drop_lines)))
    return out


@pytest.mark.parametrize("host,pid,keep,junk", CASES, ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_rule_page(host, pid, keep, junk):
    (doc, md), = _parse(host, pid)
    assert doc.parser == "rule" and not doc.fallback
    assert doc.title
    for s in keep:
        assert s in md, s
    for s in junk:
        assert s not in md, s


@pytest.mark.parametrize("host", ART39)
def test_39_article_channels(host):
    for doc, md in _parse(host):
        assert doc.parser == "rule" and not doc.fallback
        assert md.startswith("# " + doc.title)
        assert len(md) > 200
        for s in ART39_JUNK:
            assert s not in md, s


def test_zysjonline_articles_template():
    from conftest import fixture_rows
    from extract import parsers
    from extract.clean import clean_markdown
    page = [p for p in fixture_rows("zysjonline.com") if p[0] == 4178045]
    assert page
    doc = parsers.parse("zysjonline.com", page[0][1], page[0][2])
    md = clean_markdown(doc.markdown, doc.drop_lines)
    assert doc.parser == "rule" and doc.title == "逍遥汤黄芪健中汤可治疗HP感染"
    assert "幽门螺杆菌（Hp）感染率在我国高达60%" in md
