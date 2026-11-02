"""vi_health host rules on real fixture pages: rule used, real content kept (sapo, Q&A, FAQ, summaries), chrome dropped."""
import pytest

from conftest import fixture_rows
from extract.clean import clean_markdown
from extract.parsers import parse

# host, page id, sentences that must be present, strings that must be absent
CASES = [
    ("suckhoecongdongonline.vn", 424354,
     ["Vừa qua, các bác sĩ khoa Nội tổng hợp, Trung tâm Y tế huyện Thanh Thủy (Phú Thọ)",  # sapo
      "Bác sĩ cũng khuyến cáo người dân nên tẩy giun định kì 6 tháng một lần"],
     ["Đăng nhập hoặc Tạo tài khoản để tham gia bình luận", "Cùng chuyên mục"]),
    ("suckhoecongdongonline.vn", 376215,
     ["Mạng xã hội Google+ không thu hút được người dùng như kỳ vọng của Google"],
     ["Hệ điều hành bí ẩn thay thế Android"]),  # old template: related cards inside the body
    ("nhathuoclongchau.com.vn", 271714,
     ["Cong vẹo cột sống học đường là vấn đề sức khỏe trẻ ở độ tuổi đi học rất dễ gặp phải.",  # sapo
      "Học sinh không nên đeo ba lô quá nặng"],
     ["Thông tin và sản phẩm gợi ý trong bài viết chỉ mang tính chất tham khảo", "Các bài viết liên quan",
      "Kích thước chữ"]),
    ("nhathuoclongchau.com.vn", 282674,  # disease page (Wayback snapshot)
     ["Dương vật cong là một tình trạng hiếm gặp ở nam giới.", "### Dương vật cong có bình thường không?",
      "Dương vật cong nhẹ là bình thường và không ảnh hưởng đến chức năng tình dục."],
     ["Xem thêm thông tin:", "Đã kiểm duyệt nội dung"]),
    ("www.vinmec.com", 743963,  # "Hỏi bác sĩ": question and doctor answer
     ["Em đang mang thai tuần thứ 35 của thai kỳ",
      "Theo kết quả siêu âm bạn bị hở van hai lá nhẹ, hở ba lá."],
     ["Unfortunately, the content on this page is not available in English", "Để đặt lịch khám tại viện",
      "ĐỂ LẠI THÔNG TIN TƯ VẤN"]),
    ("www.vinmec.com", 737482,
     ["Thông thường, quá trình rụng trứng xảy ra khoảng 2 tuần trước khi bắt đầu kỳ kinh nguyệt"],
     ["Hình ảnh siêu âm trứng rụng có màu gì?", "Mục lục", "Unfortunately, the content on this page"]),
    ("phunusuckhoe.giadinhonline.vn", 308465,
     ["Một nghiên cứu mới đã chỉ ra rằng thực phẩm chế biến sẵn",  # sapo
      "Nghiên cứu này cũng được công bố trên eClinicalMedicine"],
     ["Giấy phép hoạt động Báo chí", "Đi công tác về, tôi tặng vợ"]),
    ("giadinhonline.vn", 109216,
     ["Ăn cá là một trong những biện pháp giúp cải thiện sức khỏe tim mạch.",
      "Thêm một muỗng bột hạt lanh vào bữa ăn mỗi ngày"],
     ["Tin nóng thế giới ngày 10/9", "Nên đọc"]),
    ("medlatec.vn", 181199,
     ["Sùi mào gà ở nam được xem là một trong những bệnh xã hội phổ biến nhất ở nam giới",  # sapo
      "Sùi mào gà ở nam là bệnh lý lây truyền qua đường tình dục do virus HPV gây ra."],
     ["Bác sĩ duyệt bài", "Từ khoá:", "16/04/2025 | Sùi mào gà âm đạo"]),
    ("hellobacsi.com", 126495,
     ["Vùng kín bị thâm đen ở tuổi dậy thì có thể là tình trạng bình thường",  # summary box
      "- Không mặc quần áo, đồ lót bó sát: Ưu tiên lựa chọn quần lót"],
     ["Đọc thêm 9 dấu hiệu dậy thì", "Đây là bài viết tổng quan cung cấp thông tin", "Thêm vào Nguồn ưu tiên"]),
    ("hellobacsi.com", 120223,
     ["Tình trạng stress oxy hóa có liên quan đến một số vấn đề sức khỏe"],  # key-takeaway box
     [">>> Bạn có thể xem thêm"]),
    ("tiemchunglongchau.com.vn", 663724,
     ["Hen suyễn và hen phế quản có giống nhau không? Đây là thắc mắc phổ biến",  # sapo
      "Hen suyễn (hay còn gọi là asthma) là một bệnh viêm mạn tính của đường hô hấp"],
     ["Miễn trừ trách nhiệm", "Nguyên nhân bị hen suyễn và cách để phòng ngừa", "Bài viết có hữu ích với bạn?"]),
    ("khoahocphothong.vn", 142656,
     ["Bé T.N.A.T 5 tuổi nhập viện Bệnh viện Nhi Đồng 1 với bệnh cảnh viêm não do thủy đậu"],
     ["Bao dien tu Tuoi Tre", "Trường Đại học Luật TP.HCM"]),
    ("youmed.vn", 764757,
     ["Seduxen là thuốc kê đơn có chứa hoạt chất chính là Diazepam."],
     ["Nguồn tham khảo / Source", "Nội dung bài viết", "Quy trình biên tập", "Biên tập bởi"]),
    ("suckhoeviet.org.vn", 564406,
     ["Thận yếu là một vấn đề sức khỏe phổ biến hiện nay",  # sapo
      "Tang phiêu diêu là tổ bọc trứng của con ngựa"],
     ["Cách sử dụng cây bách xù", "Hội Nam Y", "https://suckhoeviet.org.vn"]),
    ("pmc-ecm-healthblog.beta.pharmacity.io", 322818,
     ["Viêm kết mạc ở trẻ sơ sinh là tình trạng viêm hoặc nhiễm trùng của kết mạc"],
     [">>> Xem thêm", "Xem chi tiết", "Dược sĩ với hơn 10 năm kinh nghiệm"]),
    ("www.pharmacity.vn", 720485,
     ["vừa điều trị tăng huyết áp vừa được khuyên bổ sung canxi để bảo vệ xương."],
     ["Xem chi tiết", "Hơn 7 năm kinh nghiệm trong việc quản lý"]),
    ("tamanhhospital.vn", 570325,
     ["Trước đó một tháng, ông Tín (ngụ Lâm Đồng) thỉnh thoảng ho ra máu."],
     ["HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH", "Cập nhật lần cuối", "BÀI VIẾT CÙNG CHỦ ĐỀ", "nguồn ưu tiên trên Google"]),
    ("thaythuocvietnam.vn", 660071,  # new theme
     ["Mụn trứng cá là bệnh da liễu thường gặp nhất ở mọi lứa tuổi."],
     ["Để lại bình luận", "Nội dung bài viêt"]),
    ("thaythuocvietnam.vn", 661339,  # old theme
     ["Viêm dạ dày cấp là bệnh lý chiếm tỷ lệ rất cao trên thế giới"],
     ["Email của bạn sẽ không được hiển thị", "Những Nội Dung Cần Lưu Ý", "Bài viết cùng chủ đề", "Lượt xem"]),
    ("bachmai.gov.vn", 2651,
     ["Ung thư vú đang là gánh nặng sức khỏe toàn cầu",  # sapo
      "Chụp X.Q vú được coi là \"tiêu chuẩn vàng\" trong sàng lọc ung thư vú"],
     ["lượt xem", "Tăng huyết áp khó kiểm soát, coi chừng bệnh từ tuyến thượng thận"]),
    ("benhviennhitrunguong.gov.vn", 98259,
     ["Tính đến nay, đã có 33 ca ghép thận được thực hiện tại Bệnh viện Nhi Trung Ương"],
     ["Trang chủ »", "Bài viết liên quan", "Thời gian làm việc"]),
    ("benhvienvietduc.org", 98466,
     ["Tuổi tác: tần suất và mức độ thoái hoá khớp gối tăng theo tuổi."],
     ["Tổng đài đặt lịch khám", "Tagged in", "Chịu trách nhiệm chính"]),
    ("benhvienvietduc.org", 98714,
     ["Sau ghép tạng, người bệnh cần nghỉ ngơi đầy đủ"],
     ["sẽ tổ chức chương trình tư vấn trực tuyến"]),
]


def _page(host, pid):
    rows = {i: (url, html) for i, url, html in fixture_rows(host)}
    assert pid in rows, f"{host}: fixture page {pid} missing"
    url, html = rows[pid]
    doc = parse(host, url, html)
    return doc, clean_markdown(doc.markdown, doc.drop_lines)


@pytest.mark.parametrize("host,pid,keep,junk", CASES, ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_vi_health_rule(host, pid, keep, junk):
    doc, md = _page(host, pid)
    assert doc.parser == "rule" and not doc.fallback
    for s in keep:
        assert s in md, f"lost content: {s!r}"
    for s in junk:
        assert s not in md, f"kept chrome: {s!r}"


def test_khoahocphothong_header_not_duplicated():
    _, md = _page("khoahocphothong.vn", 134973)
    assert md.count("Theo VP UBND.TP, UBND TP vừa quyết định xếp hạng cho 13 cơ sở y tế") == 1
    assert "Khoa học" not in md.splitlines()


def test_rule_titles():
    assert _page("www.vinmec.com", 743963)[0].title.startswith("Hở 1/4 hai lá và 1/4 ba lá khi mang thai 35 tuần")
    assert _page("nhathuoclongchau.com.vn", 282674)[0].title.startswith("Dương vật cong là gì?")
