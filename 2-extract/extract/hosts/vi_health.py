"""Rules for the vi_health family (see PLAN.md, Task 7)."""
from ..rules import Rule


def _c(name: str) -> str:
    """XPath predicate: element has the exact class token `name`."""
    return f'contains(concat(" ", normalize-space(@class), " "), " {name} ")'


# "Xem thêm:" paragraph followed by a list of links to other articles.
_SEE_ALSO = '//p[starts-with(normalize-space(), "Xem thêm")][string-length(normalize-space()) < 15]/following-sibling::*[1][self::ul]'
_SEE_ALSO_LINE = r"^\**Xem thêm:?\**:?$"
_MORE_LINE = r"^(>>>|>>|»)\s*(Bạn có thể )?([Xx]em|[Đđ]ọc) thêm"  # inline "read more" link lines

RULES: dict = {}

# Sapo (lead paragraph) lives outside the body div and trafilatura drops it; related lists / comments are outside both.
RULES["suckhoecongdongonline.vn"] = Rule(
    body=(f'//div[{_c("article__sapo")}]', f'//div[{_c("article__body")}]'),
    title=f'//h1[{_c("article__title")}]',
    drop=('//table[@class="picked-relate"]',),  # old template: related-news cards inside the body
    min_chars=200,
)

# Article and "Hỏi bác sĩ" pages share #main-article (question + doctor answer are plain paragraphs inside it).
# The hidden "not available in English" modal, booking form and related posts are outside it.
RULES["www.vinmec.com"] = Rule(
    body=('//div[@id="main-article"]',),
    title=f'//h1[{_c("single-title")}]',
    drop=('//section[@id="toc-container"]', f'//div[{_c("relate-video")}]', f'//div[{_c("banner-block")}]',
          _SEE_ALSO),
    drop_lines=(_SEE_ALSO_LINE, r"^Để đặt lịch khám tại viện, Quý khách", r"^Unfortunately, the content on this page",
                r"^Ấn “Đăng kí” để theo dõi"),
    min_chars=200,
)

# Next.js template (articles /bai-viet/ and disease pages /benh/, live and Wayback snapshots alike): a grey sapo box,
# then the CMS body in div[@data-theme-element="article"] (one per section on disease pages). Approver bio, disclaimer,
# tags and related posts are siblings outside both. Disease pages render only some sections server-side: risk,
# diagnosis/treatment and prevention exist only in __NEXT_DATA__ (props.pageProps.data.*): taken via Rule.next_data.
_LC = '//div[@data-lcpr="prr-id-articles-content"]'
RULES["nhathuoclongchau.com.vn"] = Rule(
    body=(_LC + '//div[contains(@class,"rounded-xl") and contains(@class,"bg-gray-1,5")]',
          # union = document order; the FAQ block (h3 questions + answer divs) is taken whole so questions stay
          _LC + '//div[@data-theme-element="article"] | ' + _LC + '//div[h2[starts-with(normalize-space(), "Câu hỏi thường gặp")]]'),
    title=_LC + '//h1',
    drop=(_SEE_ALSO,),
    drop_lines=(_SEE_ALSO_LINE, r"^Xem thêm thông tin:"),
    min_chars=200,
    # /benh/ pages: these sections exist only in the JSON (absent on /bai-viet/ pages, so skipped there)
    next_data=("props.pageProps.data.risk", "props.pageProps.data.diagnoseAndTreaty",
               "props.pageProps.data.livingAndPreventive"),
)

# Sapo (div.post-description) sits above the image, apart from #article-body; trafilatura drops it.
RULES["phunusuckhoe.giadinhonline.vn"] = Rule(
    body=('//div[@class="post-description"]', '//div[@id="article-body"]'),
    title='//h1',
    drop=('//div[@id="article_Inview"]',),
    min_chars=200,
)

# Sapo is h2.intro next to the CMS body #explus-editor; old (2014) pages embed a "Nên đọc" related-links box.
RULES["giadinhonline.vn"] = Rule(
    body=(f'//h2[{_c("intro")}]', '//div[@id="explus-editor"]'),
    title=f'//h1[{_c("detail-title")}]',
    drop=('//div[@id="explus-editor"]/*[.//strong[normalize-space()="Nên đọc"]][string-length(normalize-space()) < 600]',),
    min_chars=200,
)

# Sapo (div.shortdescription) is dropped by trafilatura; reviewer box, keywords, comments and related lists follow.
RULES["medlatec.vn"] = Rule(
    body=(f'//div[{_c("block-posts-single")}]/div[{_c("shortdescription")}]',
          f'//div[{_c("block-posts-single")}]/div[{_c("description")}]'),
    title=f'//h1[{_c("page-title")}]',
    min_chars=200,
)

# Next.js article: each block is a div.__html__ inside div.unique-content-wrapper. trafilatura drops the summary /
# key-takeaway boxes and glues list items together; "Đọc thêm" boxes and ">>> xem thêm" lines are internal links.
RULES["hellobacsi.com"] = Rule(
    body=(f'//div[{_c("unique-content-wrapper")}]/div[{_c("__html__")}]',),
    title='//h1',
    drop=(f'//div[{_c("content-recommendation")}][starts-with(normalize-space(), "Đọc thêm")]',),
    drop_lines=(_MORE_LINE, _SEE_ALSO_LINE),
    min_chars=200,
)

# Next.js article: sapo div[@data-component="short-description"] precedes #content-article; trafilatura drops it.
RULES["tiemchunglongchau.com.vn"] = Rule(
    body=('//div[@data-component="short-description"]', '//div[@id="content-article"]'),
    title='//h1',
    drop=(_SEE_ALSO,),
    drop_lines=(_SEE_ALSO_LINE,),
    min_chars=200,
)

# trafilatura duplicates the header (category, title, sapo) and appends the "related news" rail; take sapo + body.
RULES["khoahocphothong.vn"] = Rule(
    body=(f'//p[{_c("sc-longform-header-sapo")}]', f'//div[{_c("c-news-detail")}]//div[@data-role="content"]'),
    title='//h1',
    min_chars=150,
)

# WordPress-on-Tailwind article body (div.prose). Citation <sup> carry hidden source popups that end up glued to
# headings/sentences; the TOC, editor box and the sources/editorial-policy block are chrome.
RULES["youmed.vn"] = Rule(
    body=(f'//section[{_c("article-content")}]/article/div[{_c("prose")}]',),
    title='//h1',
    drop=('//div[@id="ez-toc-container"]', '//sup[@data-id]', f'//div[{_c("not-prose")}][starts-with(normalize-space(), "Biên tập bởi")]'),
    drop_lines=(r"^Video chia sẻ thông tin", r"^Biên tập bởi:"),
    min_chars=200,
)

# MasterCMS: sapo div.article-detail-desc + body div.__MASTERCMS_CONTENT; inline "related" tables and the
# source-URL line are chrome; the right rail of "Hoạt động hội" news is outside both.
RULES["suckhoeviet.org.vn"] = Rule(
    body=(f'//div[{_c("article-detail-desc")}]', f'//div[{_c("__MASTERCMS_CONTENT")}]'),
    title=f'//h1[{_c("article-detail-title")}]',
    drop=(f'//table[{_c("__MB_ARTICLE_A")}]', f'//div[{_c("article-detail-source")}]'),
    drop_lines=(r"^Xem thêm:",),
    min_chars=200,
)

# Pharmacity blog (beta CMS host and the live site share the template): body #pmc-content-blog-detail; author bios,
# "related posts" teasers and the pharmacist card come after it.
_PMC = Rule(
    body=('//div[@id="pmc-content-blog-detail"]',),
    title='//h1',
    drop_lines=(_MORE_LINE, r"^Xin lưu ý: Thông tin trong bài viết chỉ mang tính chất tham khảo"),
    min_chars=200,
)
for _h in ("pmc-ecm-healthblog.beta.pharmacity.io", "www.pharmacity.vn"):
    RULES[_h] = _PMC

# Body is the div.post_info.box_detail_post without the title; the hospital address block (content_insert),
# "last updated / share" bar and the Google-preferred-source how-to sit at its end.
_TA_POST = f'//div[{_c("post_info")} and {_c("box_detail_post")}]'
RULES["tamanhhospital.vn"] = Rule(
    body=(_TA_POST + f'[not(div[{_c("title")}])]',),
    title=_TA_POST + f'/div[{_c("title")}]',
    drop=(f'//div[{_c("content_insert")}]', _TA_POST + '/div[contains(., "Cập nhật lần cuối")]',
          _TA_POST + '/div[contains(., "nguồn ưu tiên trên Google")]', '//div[@id="ez-toc-container"]'),
    drop_lines=(r"^Để đặt lịch thăm khám, tư vấn các vấn đề về sức khỏe tại Hệ thống Bệnh viện Đa khoa Tâm Anh",),
    min_chars=200,
)

# Two WordPress themes. New: div.content_single/div.content-post. Old: #single-content, which also holds the
# category tree, view counters, tags, related lists and comments.
_TT = '//div[@id="single-content"]'
RULES["thaythuocvietnam.vn"] = Rule(
    body=(f'//div[{_c("content_single")}]/div[{_c("content-post")}] | ' + _TT,),
    title=f'//h1[{_c("title_pages")}] | //div[@id="single-title"]/h1',
    drop=(_TT + '/div[@id="tree-cat" or @id="single-title" or @id="tag-post" or @id="comment-wrap"]',
          _TT + f'/div[{_c("date-post")} or {_c("tptn_counter")} or {_c("post-views")} or contains(@class, "related")]',
          _TT + '/h3[contains(@class, "related")]', '//div[@id="toc_container"]'),
    min_chars=200,
)

# Sapo div.content-excerpt + body div.content-editor; the "latest news" rail and dates list are outside.
_BM = '//div[@id="content-post"]'
RULES["bachmai.gov.vn"] = Rule(
    body=(_BM + f'/div[{_c("content-excerpt")}]', _BM + f'/div[{_c("content-editor")}]'),
    title=_BM + '/h3',
    min_chars=200,
)

# WordPress: div.entry-content (date line + body); breadcrumb, category, related carousel and sidebar are outside.
RULES["benhviennhitrunguong.gov.vn"] = Rule(
    body=(f'//div[@id="primary"]/div[{_c("entry-content")}]',),
    title=f'//h1[{_c("entry-title__h1")}]',
    drop=(f'//div[{_c("entry-post_info")}]',),
    min_chars=200,
)

# div.news-primary-content holds h1 + date + body + the hospital contact table, tags and "hot news" box.
_VD = '//div[@class="news-primary-content"]'
RULES["benhvienvietduc.org"] = Rule(
    body=(_VD,),
    title=_VD + '/h1',
    drop=(_VD + '/h1', _VD + f'/p[{_c("news-details-date")}]', _VD + '//table[contains(., "Tổng đài đặt lịch khám")]',
          _VD + f'/div[{_c("news-detail-bot")} or {_c("news-detail-more")}]',
          _VD + '/div[not(@class)][not(.//p or .//img or .//table)]'),  # pasted event promos (one div per line)
    drop_lines=(r"^Bệnh viện Hữu nghị Việt Đức là địa chỉ thăm khám", r"^Tagged in:"),
    min_chars=200,
)
