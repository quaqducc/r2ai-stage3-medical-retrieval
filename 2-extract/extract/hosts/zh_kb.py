"""Rules for the zh_kb family (see PLAN.md, Task 7): Chinese medical knowledge bases, TCM books, health portals."""
from ..rules import Rule

RULES: dict = {}

# zysjonline.com: TCM book chapters. Body = chapter h1 + #chapter-content; the "本篇方剂（条文出处）" block
# (citations of the same formula in other books), prev/next nav and download dialog are outside it.
# /articles/misc/ pages (699 docs) use a different template: article > h1 + div.space-y-4 body.
RULES["zysjonline.com"] = Rule(
    body=('//div[@id="chapter-content"]/preceding-sibling::h1', '//div[@id="chapter-content"]',
          '//article[div[contains(@class,"space-y-4")]]/header/h1', '//article/div[contains(@class,"space-y-4")]'),
    title='//div[@id="chapter-content"]/preceding-sibling::h1|//article[div[contains(@class,"space-y-4")]]/header/h1',
    min_chars=20,  # many chapters are a few lines long
)

# MediaWiki sites (zhongyibaodian.net, www.a-hospital.com). trafilatura drops single short paragraphs
# (list-like "＊..." lines, paragraphs with red links), so take #bodyContent and strip MediaWiki chrome.
_SEE_ALSO = ("参看", "参见", "外部链接", "参考来源", "参考资料", "相关条目", "症状查询专题", "更多医学百科条目")
_SEE_ALSO_H2 = "//div[@id='bodyContent']/h2[" + " or ".join(
    f"normalize-space()='{t}'" for t in _SEE_ALSO) + " or starts-with(normalize-space(),'健康问答网关于')]"
_MEDIAWIKI = Rule(
    body=('//h1[@id="firstHeading"]', '//div[@id="bodyContent"]'),
    title='//h1[@id="firstHeading"]',
    drop=(
        # the list/table under a see-also heading first (located via the h2), then the h2 itself
        _SEE_ALSO_H2 + "/following-sibling::*[1][self::ul or self::ol or self::table"
                       " or self::div[contains(@class,'references')]]",
        _SEE_ALSO_H2,
        '//div[@id="jump-to-nav"]', '//div[@id="contentSub"]', '//div[@class="wz-left"]', '//div[@class="wz-right"]',
        '//table[@id="toc"]', '//div[@id="catlinks"]', '//table[contains(@class,"navbox")]', '//table[@class="nav"]',
        '//div[contains(@class,"references-small")]', '//sup[contains(@class,"reference")]',
        '//table[@id="RSSMainBody"]', '//div[@id="fromlink"]', '//div[contains(@class,"bdsharebuttonbox")]',
        '//table[contains(@class,"msg-table")]', '//div[contains(@class,"thumb")]', '//div[@class="printfooter"]',
    ),
    drop_lines=(r"^按症状名称开头字母查询症状", r"^A \| B \| C \|"),
    min_chars=30,
)
for _h in ("zhongyibaodian.net", "www.a-hospital.com"):
    RULES[_h] = _MEDIAWIKI

# *.39.net article channels: one shared article template (h1 + 核心提示 summary + art_content), plus the old
# 2010 layout (.art_box: h1 + p.summary + #contentText). Listed explicitly: ask.39.net belongs to zh_qa,
# jbk.39.net (disease KB) has its own template below.
_READ_MORE = ('//div[contains(@class,"art_content") or @id="contentText"]/p[string-length(normalize-space())<20'
              ' and (contains(.,"相关阅读") or contains(.,"推荐阅读"))]')  # lines are indented with U+3000
_ART39 = Rule(
    body=('//div[@class="art_left"]/h1', '//div[@class="art_left"]/div[contains(@class,"art_summary")]',
          '//div[@class="art_left"]/div[contains(@class,"art_content")]',
          '//div[@class="art_box"]/h1', '//div[@class="art_box"]/p[@class="summary"]', '//div[@id="contentText"]'),
    title='//div[@class="art_left"]/h1|//div[@class="art_box"]/h1',
    drop=('//div[contains(@class,"art_show")]', '//div[contains(@class,"art_page")]',
          '//div[contains(@class,"art_rec")]', '//div[contains(@class,"hot_art")]',
          _READ_MORE + "/following-sibling::p[a][normalize-space(translate(.,'　',''))=normalize-space(.//a)]",  # U+3000 indent
          _READ_MORE),
    drop_lines=(r"^[（(](责任|实习)编辑[：:].*[)）]$", r"^\**39健康网\( ?www\.39\.net\)专稿", r"^举报/反馈$",
                r"^[（(]以上内容仅授权39健康网", r"^未经作者允许授权", r"^>>点击查看",
                r"^关注“39减肥健身学院”微信公号", r"^想要健康瘦下来吗？", r"^【送福利：",
                r"^责任编辑[：:]", r"^更多.{0,10}最新资讯，请关注39", r"^核心提示[：:]$",
                r"^更多.{0,20}尽在"),
    min_chars=80,
)
for _h in ("baby.39.net", "care.39.net", "heart.39.net", "fk.39.net", "nk.39.net", "cancer.39.net", "tnb.39.net",
           "woman.39.net", "food.39.net", "ek.39.net", "man.39.net", "shen.39.net", "gan.39.net", "wei.39.net",
           "sj.39.net", "fitness.39.net", "gk.39.net", "pf.39.net", "gc.39.net", "byby.39.net"):
    RULES[_h] = _ART39

# jbk.39.net disease KB. Disease page: name + first disease_box (summary + 是否医保/科室/症状... facts); the rest
# (similar diseases, doctors, drug reviews, ads) is junk. Exam page (/jiancha/): h1 + intro + facts + sections.
RULES["jbk.39.net"] = Rule(
    body=('//div[@class="disease"]', '//div[@class="list_left"]/div[contains(@class,"disease_box")][1]',
          '//div[contains(@class,"ss_det")]//h1', '//div[@id="intro"]', '//ul[@class="infolist"]',
          '//div[contains(@class,"lbox") and contains(@class,"catalogItem")]'),
    title='//div[@class="disease"]//h1|//div[contains(@class,"ss_det")]//h1',
    drop=('//p[@class="disease_title"]', '//p[@class="information_l"]/a', '//div[@id="intro"]/i',
          '//ul[@class="infolist"]/li[contains(@class,"data") or .//i[@class="key"]]', '//a[@class="more"]',
          '//div[contains(@class,"lbox")][div[@class="listBox"]]'),  # related disease/symptom link lists
    drop_lines=(r"^- 在线购药：$", r"^- 科室：$"),
    min_chars=40,
)

# test.pmphai.com (人卫临床助手): case / disease entries; paywalled sections show only "此内容为收费内容".
_PAID = '//div[@class="gu_det_left_con_main"][normalize-space()="此内容为收费内容"]'
RULES["test.pmphai.com"] = Rule(
    body=('//div[@class="gu_det_left_top"]', '//div[@id="content"]'),
    title='//div[@class="gu_det_left_top"]',
    drop=(_PAID + '/preceding-sibling::div[1][@class="gu_det_left_con_h1"]', _PAID),
    min_chars=30,
)

# www.msdmanuals.cn (Next.js): the server-rendered HTML holds the full topic (headings, lists, sections);
# the __NEXT_DATA__ text is fragmented at every link and mixed with tooltip texts of other topics.
RULES["www.msdmanuals.cn"] = Rule(
    body=('//div[contains(@class,"TopicHead_tablebox__cell")]//h1',
          '//div[contains(@class,"Topic_topicContainer__")]/div[contains(@class,"TopicPara_topicPara")]',
          '//div[contains(@class,"TopicMainContent_content")]'),
    title='//div[contains(@class,"TopicHead_tablebox__cell")]//h1',
    drop_lines=(r"^(Procedure|Image|Photo|Video|Illustration)s? (by|courtesy of) ", r"^©"),  # figure credits
    min_chars=150,
)

# www.wujue.com: article title + .art-text; drop the page-number row.
RULES["www.wujue.com"] = Rule(
    body=('//div[@class="main_articletitle"]', '//div[@class="art-text"]'),
    title='//div[@class="main_articletitle"]',
    drop=('//div[@class="art-text"]//p[.//a[contains(.,"下一页") or contains(.,"上一页")]]',),
    min_chars=40,
)

# jb39.com disease overview: title, aliases, intro, per-section excerpts (the site itself truncates them with
# "……"; full sections live on sub-pages). Union keeps document order; related/search link lists are dropped.
RULES["jb39.com"] = Rule(
    body=('//h1[contains(@class,"post-title")]',
          '//div[@class="post-jibing"]/*[self::div[contains(@class,"post-bieming")] or self::div[@class="jb-body"]'
          ' or self::h2[contains(@class,"jb-h2")] or self::p[contains(@class,"jb-zy")]]'),
    title='//h1[contains(@class,"post-title")]',
    min_chars=40,
)

# www.zydcd.com (中医大辞典): dictionary entries, case records, articles share one panel layout.
RULES["www.zydcd.com"] = Rule(
    body=('//div[contains(@class,"panel-danger")]/div[@class="panel-heading"]',
          '//div[contains(@class,"panel-danger")]/div[contains(@class,"info_txt")]'),
    title='//div[contains(@class,"panel-danger")]/div[@class="panel-heading"]',
    min_chars=20,
)

# health.people.com.cn: old (.articleCont) and new (.rm_txt) article templates.
RULES["health.people.com.cn"] = Rule(
    body=('//div[@class="articleCont"]/div[@class="title"]', '//div[@class="articleCont"]/div[@class="artDet"]',
          '//div[contains(@class,"rm_txt")]/div[contains(@class,"col-1")]/h1', '//div[@id="rm_txt_zw"]'),
    title='//div[@class="articleCont"]/div[@class="title"]/h2|//div[contains(@class,"rm_txt")]/div[contains(@class,"col-1")]/h1',
    drop=('//div[@class="articleCont"]/div[@class="title"]/ul', '//div[contains(@class,"zdfy")]',
          '//div[contains(@class,"box_pic")]'),
    min_chars=80,
)

# qihuangzhishu.com: TCM book chapters (h1 + .cbd); .cbd also holds a related-term list and download links.
RULES["qihuangzhishu.com"] = Rule(
    body=('//h1[contains(@class,"ch1")]', '//div[@class="cbd"]'),
    title='//h1[contains(@class,"ch1")]',
    drop=('//ul[contains(@class,"zzhj")]',),
    drop_lines=(r"^1000余本中医古籍txt电子书免费下载$", r"^下载《.*》 ?电子书打不开？$", r"^下载所有中医书籍$"),
    min_chars=20,
)
