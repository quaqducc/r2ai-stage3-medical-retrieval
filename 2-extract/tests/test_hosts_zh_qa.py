import pytest

from conftest import fixture_rows
from extract import clean, parsers

# host -> (sentences that must appear in some fixture page, strings that must appear in none)
CASES = {
    "www.120ask.com": (
        ["根据您的情况,建议您到医院做个血HCG检查,比较准确.",  # doctor answer (lives in .b_answerbox)
         "脾肾两助丸主要具有健脾益气"],
        ["药品库", "我要投诉", "微信扫一扫", "请选择投诉理由", "赠送医生锦旗"],
    ),
    "ask.39.net": (
        ["你好，专业的妇科大夫是可以看出来的，不必担心",  # second (non-featured) answer, dropped by trafilatura
         "牙齿裂开一般不需要直接拔除"],
        ["其他人还在看", "声明：在39问医生平台", "症状起因", "擅长：", "举报", "精选回答", "找医生"],
    ),
    "www.cnkang.com": (
        ["建议肿瘤科就诊", "父母要洁身自爱，注意个人卫生。"],
        ["专题推荐", "相关文章", "上一篇", "温馨提示：以上资料仅供参考", "热点排行", "核心提示"],
    ),
    "www.familydoctor.com.cn": (
        ["II型糖尿病老年患者中，恶性肿瘤发生率高达28.35%", "少量多餐，空腹过久会导致饥饿性酮症"],
        ["专家答疑", "热门文章", "责任编辑", "相关文章推荐"],
    ),
    "baidianfeng.familydoctor.com.cn": (
        ["在白癜风患者的视网膜异常中，多为局灶性色素增多"],
        ["专家答疑", "热门文章", "责任编辑"],
    ),
    "qy.familydoctor.com.cn": (
        ["切忌偏食，吃得越杂越好。"],
        ["擅长：", "家庭医生在线专稿", "皮肤病医院"],
    ),
    "www.youlai.cn": (
        ["洗牙主要是用来清除牙齿和牙龈之间的软垢", "流感的全身症状比较明显"],
        ["手机浏览", "用手机扫描二维码", "相关推荐", "医生推荐", "医生作品"],
    ),
    "article.iiyi.com": (
        ["四、通过本次手术，对凶险性前置胎盘手术的麻醉体会如下：",  # heading in a bare <span>, dropped by trafilatura
         "1.双剑博，甲状腺良性结节微创治疗进展"],
        ["版权声明", "本站所注明来源", "分享到", "意见反馈", "zhoumeifang@120.net"],
    ),
}


@pytest.mark.parametrize("host", sorted(CASES))
def test_zh_qa_fixture(host):
    rows = fixture_rows(host)
    assert rows
    must, junk = CASES[host]
    texts = []
    for _, url, html in rows:
        doc = parsers.parse(host, url, html)
        assert doc.parser == "rule" and not doc.fallback, (url, doc.parser)
        md = clean.clean_markdown(doc.markdown, doc.drop_lines)
        for j in junk:
            assert j not in md, (url, j)
        texts.append(md)
    for s in must:
        assert any(s in t for t in texts), s
