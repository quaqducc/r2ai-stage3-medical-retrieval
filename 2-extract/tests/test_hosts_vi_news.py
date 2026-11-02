"""vi_news host rules on committed fixture pages (two real pages per host)."""
import re

import lxml.html
import pytest

from conftest import fixture_rows
from extract import clean, parsers

# host -> (content sentence seen in one fixture page, chrome string present on the pages that must not be extracted)
CASES = {
    "suckhoedoisong.vn": ("SKĐS - Trong mùa nắng nóng, chúng ta thường bài tiết nhiều mồ hôi hơn", "Xem thêm"),
    "thanhnien.vn": ("Hỏi: Tôi đang rất đau khổ khi biết mình có thai ngoài tử cung.", "LƯU Ý NHỎ"),
    "tuoitre.vn": ("Cục An toàn thực phẩm, Bộ Y tế vừa có văn bản khẩn gửi Sở Y tế Đà Nẵng", "Tối đa"),
    "laodong.vn": ("Hạt chia, hạt hướng dương, hạt hạnh nhân cung cấp chất béo tốt và protein",
                   "Bạn nghĩ gì về nội dung này"),
    "baohaiphong.vn": ("Trước tình trạng nguy hiểm của cha, cậu con trai 18 tuổi nhận hiến gan.", "Bài liên quan"),
    "baonghean.vn": ("Trong kho tàng ẩm thực của người Việt, trứng gà là thực phẩm quen thuộc", "POWERED BY"),
    "baodanang.vn": ("(QNO) - Ngày 12 - 13.5, Công an huyện Phú Ninh ra quân kiểm tra", "POWERED BY"),
    "hanoimoi.vn": ("Mảnh xương cá dài 4cm đâm thủng ruột non", "Đừng bỏ lỡ"),
    "vietnamnet.vn": ("Để trái cây chín đều, lên màu đẹp thương lái và nhà vườn thường dùng các hóa chất",
                      "Cơ quan chủ quản"),
    "baocantho.com.vn": ("Sony đã bắt tay với các nhà nghiên cứu của Viện Max Planck ở Đức", "Đọc nhiều"),
    "baoangiang.com.vn": ("Kết quả cho thấy những phụ nữ cho con bú trong 6 - 12 tháng có rủi ro bị tiểu đường",
                          "Mọi phản ánh, ý kiến"),
    "vov.vn": ("VOV.VN - Khi cơ thể thiếu hụt vitamin C, các quá trình sinh hóa sẽ bị đình trệ", "Google Search"),
    "phuyen.baodaklak.vn": ("10 KHUYẾN CÁO PHÒNG CHỐNG CÚM A/H1N1", "Cơ quan chủ quản"),
    "nhandan.vn": ("đơn vị vừa thực hiện thành công liên tiếp 2 ca phẫu thuật nội soi cắt ruột thừa", "Tin liên quan"),
    "baovinhlong.com.vn": ("Qua test nhanh COVID-19 ra toàn dân tại xã Phú Quới và Hòa Phú", "Thông tin bạn đọc"),
    "www.qdnd.vn": ("Khoa Sản của bệnh viện vừa mổ đẻ thành công cho sản phụ V.T.Q", "Mời bạn đọc vào chuyên mục"),
    "www.vietnamplus.vn": ("AstraZeneca đã đạt thỏa thuận mua Amolyt Pharma", "Lưu bài viết"),
    "baogialai.com.vn": ("Bé trai 8 tuổi suýt mất vành tai phải do chó cắn.", "Xem link nguồn"),
    "tienphong.vn": ("TPO - Dù được tích cực cấp cứu, song một sản phụ cùng thai nhi đã tử vong", "Đọc nhiều"),
    "baoquangninh.vn": ("Bệnh nhân Li Zichao, một trong hai người đang điều trị tại Bệnh viện Chợ Rẫy", "từ khóa"),
    "baolangson.vn": ("Trong 8 tháng đầu năm 2025, trên địa bàn tỉnh có 677/4.812 trẻ sinh ra là con thứ 3",
                      "Tất cả chuyên mục"),
    "www.sggp.org.vn": ("Sau phẫu thuật, sản phụ đã qua cơn nguy hiểm và đang được chăm sóc.", "Từ khóa"),
    "baotayninh.vn": ("Trung tâm y tế huyện Bến Lức chỉ đạo các Trạm y tế xã", "Liên kết website"),
    "baothanhhoa.vn": ("Các bác sĩ khuyến cáo người dân tuyệt đối không xông than trong phòng kín",
                       "Xin vui lòng gõ tiếng Việt có dấu"),
    "baophutho.vn": ("Chiều 29/1, tại phường Việt Trì, Bệnh viện Mắt Hà Nội - Phú Thọ đã tổ chức lễ khai trương", "{title}"),
    "vnexpress.net": ("bác sĩ Vũ Sơn Tùng tiếp nhận bệnh nhân Trọng, 43 tuổi", "BVĐK TÂM ANH"),
}

# Pages whose whole article (title included) is an encrypted blob decoded by page JS: no rule can extract them.
ENCRYPTED = {("baophutho.vn", 81080)}


def _visible(html):
    """Page text with entities decoded (some hosts write every Vietnamese letter as &#...;)."""
    return " ".join(lxml.html.fromstring(re.sub(r"</(?:html|body)\s*>", "", html, flags=re.I)).text_content().split())


def _docs(host):
    rows = fixture_rows(host)
    assert rows, f"no fixture rows for {host}"
    out = []
    for pid, url, html in rows:
        doc = parsers.parse(host, url, html)
        out.append((pid, html, doc, clean.clean_markdown(doc.markdown, doc.drop_lines)))
    return out


@pytest.mark.parametrize("host", sorted(CASES))
def test_vi_news_rule(host):
    content, junk = CASES[host]
    docs = _docs(host)
    for pid, html, doc, md in docs:
        if (host, pid) not in ENCRYPTED:
            assert doc.parser == "rule", (host, pid, doc.parser)
            assert md.startswith("# "), (host, pid, md[:80])  # the article title leads the document
        assert junk not in md, (host, pid, junk)
    assert any(content in md for _, _, _, md in docs), (host, content)
    assert any(junk in _visible(html) for _, html, _, _ in docs), f"junk string {junk!r} not on {host} pages: weak test"


def test_vi_news_sapo_kept():
    # trafilatura dropped these lead paragraphs (sapo); the rules put them right after the title.
    for host, sapo in (("suckhoedoisong.vn", "SKĐS - Để đề phòng sẹo xấu"),
                       ("thanhnien.vn", "Ngày 20.8, tin từ Bệnh viện đa khoa tỉnh Đồng Tháp"),
                       ("www.qdnd.vn", "Chiều 26-6, TS Nguyễn Đình Hưng")):
        mds = [md for _, _, _, md in _docs(host)]
        assert any(md.splitlines()[2].startswith(sapo) for md in mds), host


def test_vi_news_no_duplicate_title():
    for host in ("vnexpress.net", "baothanhhoa.vn"):
        for pid, _, doc, md in _docs(host):
            heads = [l for l in md.splitlines() if l.startswith("# ")]
            assert len(heads) == 1, (host, pid, heads)


def test_vi_news_encrypted_blob_never_extracted():
    for host in ("baophutho.vn", "baothanhhoa.vn"):
        for pid, _, doc, md in _docs(host):
            assert not any(len(l) > 120 and " " not in l for l in md.splitlines()), (host, pid)
            assert "{title}" not in md and "{publish}" not in md
