"""Rules for the vi_news family: Vietnamese newspapers, health sections (see PLAN.md, Task 7).

Most hosts run on a handful of CMS templates, so one Rule serves several hosts. Each rule reads
title -> sapo (lead) -> body, and drops inline related-news boxes, ads, tags, comments and share bars.
Hosts not listed here (baochinhphu, dantri, vov2, baoquangtri, mega/special.vietnamplus)
are extracted well by trafilatura.
"""
from dataclasses import replace

from ..rules import Rule


def _cls(name: str) -> str:
    """XPath predicate: element has the exact class token `name`."""
    return f'contains(concat(" ", normalize-space(@class), " "), " {name} ")'


_BLOB = r"^[A-Za-z0-9+/=]{120,}$"  # dhcn CMS (baothanhhoa, baophutho) ships some bodies encrypted, decoded by page JS

# Common chrome lines left by several templates.
_COMMON_LINES = (
    r"^Bình luận( \(\d+\))?:?$", r"^Ý kiến \(\d*\)$", r"^(Bài liên quan|Tin liên quan|Đọc tiếp|Xem thêm|Đọc thêm)$",
    r"^(Cỡ chữ|Mặc định|Nổi bật|Đừng bỏ lỡ)$", r"^Gửi bình luận$",
)

RULES: dict = {}

# VCCorp "detail" template: <h1 data-role=title>, <* data-role=sapo>, <div data-role=content>.
_VCC = Rule(
    body=('//h1[@data-role="title" or .//*[@data-role="title"]]', '//*[@data-role="sapo"]', '//div[@data-role="content"]'),
    title='//h1[@data-role="title" or .//*[@data-role="title"]]',
    drop=('//div[@data-role="content"]//*[starts-with(@type, "Related")]',      # RelatedNewsBox / RelatedOneNews
          '//div[@data-role="content"]//article[contains(@class, "story")]',  # thanhnien old "tin liên quan"
          '//div[@data-role="content"]//*[contains(@class, "VCObjectBoxRelatedNews")]',
          '//div[@data-role="content"]//*[contains(@class, "readmore-body-box")]',
          '//h1//div[@data-role="title-bottom"]'),  # an empty block inside h1 breaks the heading
    drop_lines=_COMMON_LINES + (r"^Tối đa: ?$", r"^1500 ký tự", r"^Hiện chưa có bình luận nào"),
    min_chars=200,
)
for _h in ("suckhoedoisong.vn", "thanhnien.vn", "tuoitre.vn"):
    RULES[_h] = _VCC

# "zce" CMS (Nhân Dân, VietnamPlus, Tiền Phong, SGGP, ...): h1.cms-title, .cms-desc (sapo), div.cms-body.
_ZCE = Rule(
    body=('//h1[contains(@class, "cms-title")]', '//*[contains(@class, "cms-desc")]', '//div[contains(@class, "cms-body")]'),
    title='//h1[contains(@class, "cms-title")]',
    drop=('//div[contains(@class, "cms-body")]//*[contains(@class, "box-related")]',
          '//div[contains(@class, "cms-body")]//*[contains(@class, "related-news") or contains(@class, "related-topic")]',
          '//div[contains(@class, "cms-body")]//div[contains(@class, "article__tag") or contains(@class, "article__meta")]',
          '//div[contains(@class, "cms-body")]//article[contains(@class, "story")]',
          f'//div[contains(@class, "cms-body")]//*[{_cls("ads")} or contains(@class, "banner") or contains(@class, "social")]',
          # baothanhhoa repeats title and sapo inside the body
          '//div[contains(@class, "cms-body")]/*[contains(@class, "pTitle") or contains(@class, "pHead")]'),
    drop_lines=_COMMON_LINES + (_BLOB, r"^\*{0,2}\[links\(\)\]\*{0,2}$",
                                r"^Xin vui lòng gõ tiếng Việt có dấu$", r"^Xem link nguồn$"),
    min_chars=200,
)
for _h in ("nhandan.vn", "www.vietnamplus.vn", "baogialai.com.vn", "tienphong.vn", "baoquangninh.vn",
           "www.sggp.org.vn"):
    RULES[_h] = _ZCE
RULES["baothanhhoa.vn"] = replace(_ZCE, meta_fallback=True)  # dhcn-encrypted bodies: keep title + og:description

# ONECMS (by NEKO): div.c-news-detail > div.b-maincontent > .entry (longform header: category, h1, meta, sapo).
_ONE_BODY = f'//div[contains(@class, "c-news-detail")]//*[{_cls("entry")}]'
_ONECMS = Rule(
    body=('//h1[contains(@class, "c-detail-head__title")]',  # old layout: title and sapo outside .entry
          '//div[contains(@class, "b-maincontent")]/p[contains(@class, "desc")]', _ONE_BODY),
    title=f'{_ONE_BODY}//h1 | //h1[contains(@class, "c-detail-head__title")]',
    drop=(f'{_ONE_BODY}//*[contains(@class, "sc-longform-header-cate")]',
          f'{_ONE_BODY}//*[contains(@class, "sc-longform-header-meta")]',
          f'{_ONE_BODY}//*[contains(@class, "banner") or contains(@class, "Ads")]',
          f'{_ONE_BODY}//table[.//th[contains(., "TIN LIÊN QUAN")]]',
          f'{_ONE_BODY}//*[contains(@class, "c-box")]'),
    drop_lines=_COMMON_LINES,
    min_chars=200,
)
for _h in ("baohaiphong.vn", "baonghean.vn", "baodanang.vn", "hanoimoi.vn"):
    RULES[_h] = _ONECMS

# Same CMS as baotayninh: div.content-article > h1.name + div#content-detail.
_CONTENT_ARTICLE = Rule(
    body=('//div[contains(@class, "content-article")]/h1', '//div[@id="content-detail"]'),
    title='//div[contains(@class, "content-article")]/h1',
    drop=('//div[@id="content-detail"]//*[contains(@class, "relate")]',
          f'//div[@id="content-detail"]//*[{_cls("ads")}]'),
    drop_lines=_COMMON_LINES + (r"^https?://\S+$",),  # bare source URL under re-posted articles
    min_chars=200,
)
for _h in ("baotayninh.vn", "baolangson.vn"):
    RULES[_h] = _CONTENT_ARTICLE

# WordPress tagdiv theme imports.
RULES["phuyen.baodaklak.vn"] = Rule(
    body=('(//div[contains(@class, "detail-main-title")]//h1)[1]', '//div[contains(@class, "td-post-content")]'),
    title='(//div[contains(@class, "detail-main-title")]//h1)[1]',
    min_chars=200,
)
RULES["baovinhlong.com.vn"] = Rule(
    body=('//h1[contains(@class, "head_title")]', '//div[contains(@class, "td-post-content")]'),
    title='//h1[contains(@class, "head_title")]',
    drop_lines=(r"^#* ?Thông tin bạn đọc$", r"^Đóng Lưu thông tin$", r"^\[links\(\)\]$"),
    min_chars=200,
)

_LD = '//article[contains(@class, "article-detail")]'
RULES["laodong.vn"] = Rule(
    body=(f'{_LD}//h1', f'{_LD}//div[contains(@class, "chappeau")]', f'{_LD}//div[{_cls("art-body")}]'),
    title=f'{_LD}//h1',
    drop=(f'{_LD}//div[{_cls("art-body")}]//*[contains(@class, "tin-lien-quan") or contains(@class, "lst-advs")'
          ' or contains(@class, "stories-box")]',),
    drop_lines=_COMMON_LINES,
    min_chars=150,
)

RULES["vietnamnet.vn"] = Rule(
    body=('//h1[contains(@class, "content-detail-title")]', '//div[contains(@class, "content-detail-sapo")]',
          '//div[@id="maincontent"]'),
    title='//h1[contains(@class, "content-detail-title")]',
    drop=('//div[@id="maincontent"]//article[contains(@class, "ck-cms-wiki-news-full")]',  # inline related
          '//div[@id="maincontent"]//*[contains(@class, "ck-cms-insert-news") or contains(@class, "inner-article")]',
          '//div[@id="maincontent"]//*[contains(@class, "banner")]'),
    min_chars=200,
)

RULES["baocantho.com.vn"] = Rule(
    body=('//div[@id="colcate1"]/h1[contains(@class, "fw700")]', '//div[@id="newscontents"]'),
    title='//div[@id="colcate1"]/h1[contains(@class, "fw700")]',
    min_chars=200,
)

RULES["baoangiang.com.vn"] = Rule(
    body=('//div[contains(@class, "col-left")]//div[contains(@class, "detail-title")]',
          '//div[contains(@class, "col-left")]//div[contains(@class, "detail-desc")]', '//div[@id="newscontents"]'),
    title='//div[contains(@class, "col-left")]//div[contains(@class, "detail-title")]',
    min_chars=200,
)

# vov.vn: anchor every selector on an article that has div.text-long, so sponsored/video layouts fall back.
_VOV = '//article[contains(@class, "detail")][.//div[contains(@class, "text-long")]]'
RULES["vov.vn"] = Rule(
    body=(f'{_VOV}//div[contains(@class, "article-title")]', f'{_VOV}//div[contains(@class, "article-summary")]',
          f'{_VOV}//div[contains(@class, "text-long")]'),
    title=f'{_VOV}//div[contains(@class, "article-title")]',
    drop=(f'{_VOV}//div[contains(@class, "text-long")]//article[contains(@class, "inner-article")]',),  # related
    min_chars=200,
)

# baophutho: #dcontent holds h1.pTitle + p.pHead (sapo) + body; trafilatura duplicates and adds JS templates.
# Many pages carry only the encrypted blob (title and sapo included).
# Such pages fail the rule and fall back to the page's og:description (meta_fallback).
RULES["baophutho.vn"] = Rule(
    body=('//div[@id="dcontent"]',), title='//div[@id="dcontent"]//h1',
    drop_lines=(_BLOB, r"^\{\w+\}$", r"^#+$", r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$",
                r"^Ý kiến của bạn sẽ được biên tập", r"^\*\*baophutho\.vn\*\* .*\.\.\.$"),
    min_chars=200,    meta_fallback=True,  # encrypted pages: title + og:description (the sapo), not trafilatura's teasers
)

_QD = '//div[contains(@class, "article-detail")]'
RULES["www.qdnd.vn"] = Rule(
    body=(f'{_QD}/h1[contains(@class, "post-title")]', f'{_QD}//div[contains(@class, "post-summary")]',
          f'{_QD}//div[contains(@class, "post-content")]'),
    title=f'{_QD}/h1[contains(@class, "post-title")]',
    drop_lines=(r"^\W*Mời bạn đọc vào chuyên mục .* xem các tin, bài liên quan",),
    min_chars=200,
)

_VNE = '//article[contains(@class, "fck_detail")]'
RULES["vnexpress.net"] = Rule(
    body=('//h1[contains(@class, "title-detail")][not(ancestor::article[contains(@class, "fck_detail")])]',
          '//p[contains(@class, "description")][not(ancestor::article[contains(@class, "fck_detail")])]', _VNE),
    title='//h1[contains(@class, "title-detail")]',
    drop=('//p[contains(@class, "description")]//span[contains(@class, "location-stamp")]',
          f'{_VNE}//table[contains(., "Độc giả") and contains(., "câu hỏi")]',
          f'{_VNE}//*[contains(@class, "box_brief_info") or contains(@class, "banner")]'),
    drop_lines=(r"Độc giả (gửi|đặt) câu hỏi .* tại đây",),
    min_chars=200,
)
