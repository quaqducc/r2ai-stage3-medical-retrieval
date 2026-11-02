"""Rules for the zh_qa family (see PLAN.md, Task 7): Chinese medical Q&A forums and their article channels."""
from ..rules import Rule


def _cls(c: str) -> str:
    """XPath predicate: element has the exact class token c."""
    return f'contains(concat(" ", normalize-space(@class), " "), " {c} ")'


_YL_TITLE = '//div[starts-with(@class,"mainContent--")]/div[starts-with(@class,"title--")]'

RULES: dict = {
    # QA: question = h1 + description + follow-ups; every doctor answer lives in .b_anscont_cont inside .b_answerbox.
    # trafilatura kept the site menu (用药 药品库 …) and the complaint dialog.
    "www.120ask.com": Rule(
        body=('//h1[@id="d_askH1"]', '//div[@id="d_msCon"]', f'//div[{_cls("b_askcont_add")}]',
              f'//div[{_cls("b_answerbox")}]//div[{_cls("b_anscont_cont")}]'),
        title='//h1[@id="d_askH1"]',
        drop=(f'//a[{_cls("tousu")}]', f'//div[{_cls("b_askbot_ad1")}]'),
        min_chars=40,
    ),
    # QA: .ask_cont = question; each answer block is a div.selected (精选回答 and the other 医生回答 — trafilatura
    # dropped the latter). Drop tags, "其他人还在看", doctor bio, report links; the 疾病百科 sidebar is never selected.
    "ask.39.net": Rule(
        body=(f'//div[{_cls("ask_cont")}]', f'//div[{_cls("selected")}]'),
        title=f'//h1[{_cls("ask_tit")}]',
        drop=(f'//section[{_cls("sys-box")}]', f'//div[{_cls("txt_answer")}]', f'//p[{_cls("txt_label")}]',
              f'//p[{_cls("txt_nametime")}]', f'//div[{_cls("doc_img")}]', f'//p[{_cls("doc_sc")}]',
              f'//p[{_cls("ask_doc")}]', f'//p[{_cls("doc_time")}]', f'//span[{_cls("txt_complaint")}]',
              f'//div[{_cls("txt_yuedu")}]'),
        drop_lines=(r"^(精选回答|医生回答|网友回答|其他回答|全部回答)\(\d+\)$", r"^发病时间：不清楚$"),
        min_chars=40,
    ),
    # Articles, two templates: new /dise/ (div.detail h1 + div.detailc) and old (div.endContent h1 + div#endText).
    "www.cnkang.com": Rule(
        body=('//div[@class="detail"]/h1', f'//div[{_cls("detailc")}]', '//div[@class="endContent"]/h1',
              '//div[@id="endText"]'),
        title='//div[@class="detail"]/h1|//div[@class="endContent"]/h1',
        drop=(f'//div[{_cls("article_intro")}]',),
        drop_lines=(r"^温馨提示：以上资料仅供参考",),
    ),
    # Articles: trafilatura kept 专家答疑 / 热门文章 lists after the body.
    "www.familydoctor.com.cn": Rule(
        body=(f'//div[{_cls("article-titile")}]//h1', '//div[@id="viewContent"]'),
        title=f'//div[{_cls("article-titile")}]//h1',
        drop_lines=(r"^（责任编辑：.*）$", r"家庭医生在线（www\.familydoctor\.com\.cn）原创内容",
                    r"^家庭医生在线专稿，转载请注明"),
    ),
    # Articles (CSS-module class names with a hash suffix): kept 视频/文章/语音 tabs, QR text, 医生推荐.
    "www.youlai.cn": Rule(
        body=(_YL_TITLE, '//div[starts-with(@class,"textContent--")]'),
        title=_YL_TITLE,
    ),
    # Case articles: trafilatura dropped headings that sit in bare <span> under div.cont and doubled 版权声明.
    "article.iiyi.com": Rule(
        body=(f'//div[{_cls("article_cont")}]/div[{_cls("top")}]/h1',
              f'//div[{_cls("article_cont")}]/div[{_cls("cont")}]'),
        title=f'//div[{_cls("article_cont")}]/div[{_cls("top")}]/h1',
        drop=(f'//div[{_cls("b-mh-bto")}]', '//div[font][contains(., "版权归作者")]'),
        drop_lines=(r"^版权声明[:：]?$", r'^本站所注明来源为"爱爱医"', r"^本站所有转载文章系出于传递更多信息之目的"),
    ),
}
# Same article template on the familydoctor channel subdomains (small hosts).
for _h in ("baidianfeng.familydoctor.com.cn", "qy.familydoctor.com.cn"):
    RULES[_h] = RULES["www.familydoctor.com.cn"]
