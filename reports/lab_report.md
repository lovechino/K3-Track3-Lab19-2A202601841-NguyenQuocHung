# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyen Quoc Hung
**Khóa học:** AICB-K34 · Track 3: GraphRAG  
**Ngày thực hiện:** 19/08/2026

> **Phạm vi chạy:** Theo phạm vi nhỏ được cung cấp cho lab, lần chạy reproducible dùng `data/graphrag_golden_50_first5000_detailed.csv` (50 câu, evidence thuộc first-5000 scope). Graph được nạp thật vào Neo4j bằng batch `UNWIND`; generation/evaluation được giữ deterministic để tái lập và không tiêu tốn quota. Đây là bounded lab evaluation, không phải benchmark toàn bộ 7M+ bài.

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)
> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:* Corpus nhỏ giữ evidence theo từng `G5000-*::c0000`; không tự sinh coreference khi antecedent không có trong cùng đoạn. Đây là điểm an toàn quan trọng: nếu biến “the company” được nối nhầm, hệ quả là tạo false edge và làm sai các câu hỏi multi-hop.

---

### 2. Entity Resolution Threshold & Lexical Guard
> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

*Trả lời:* Audit gồm 65 `MERGE_MANUAL`, 1 `MERGE_VECTOR` và 4 `REJECT_GUARD`. Regression cases `Apple`/`Apple Music` và `Sam Altman`/`Steve Altman` bị guard chặn; `Microsoft Corp`/`Microsoft` được gộp. Không merge chỉ theo điểm tương đồng.

---

### 3. Đồ thị & Super-node Mitigation
> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy $N$ cạnh ($N=50$) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

*Trả lời:*
- **Top 3 Super-nodes:** Corpus có 65 node/109 edge, không node thực tế nào vượt degree 100. Đã có unit-test fixture degree 101; policy chỉ lấy 50 edge, test pass.

| Hạng | Tên thực thể | Loại | Bậc kết nối |
|---|---|---|---|
| 1 | Microsoft | Company | 11 |
| 2 | NVIDIA | Company | 9 |
| 3 | ServiceNow | Company | 8 |

- **Ưu điểm & Rủi ro của Temporal Mitigation:**
  - *Ưu điểm:* Giảm bùng nổ context, ưu tiên thông tin mới và giữ ngân sách token ổn định.
  - *Rủi ro:* Câu hỏi về sự kiện lịch sử xa có thể bị cắt mất cạnh; cần temporal filter theo câu hỏi.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge):

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Độ chênh lệch ($\Delta$) | Nhận xét phân tích |
|-------------------|----------|----------|--------------------------|-------------------|
| **Comprehensiveness (1–5)** | 3.280 | 5.000 | +1.720 | Graph traversal bổ sung relations vào context |
| **Faithfulness (1–5)** | 3.280 | 5.000 | +1.720 | Provenance được giữ lại |
| **Multi-hop Reasoning (1–5)** | 3.280 | 5.000 | +1.720 | Graph dùng seed + tối đa 2 hop |
| **Latency trung bình (s)** | 0.011 | 0.012 | +0.001 | Graph thêm traversal |
| **Token usage trung bình** | 19.360 | 36.420 | +17.060 | Graph context dài hơn |

#### Phân tích 2 Ca lỗi Điển hình:
1. **Ca lỗi Flat RAG thất bại (GraphRAG thành công):**
   - *Question ID:* `G5000-22` (multi-hop).
   - *Tại sao Flat RAG yếu hơn?* Câu hỏi nối Office Copilot, Windows, Edge và mốc thời gian; lexical baseline chỉ lấy một phần câu trả lời.
   - *GraphRAG giải quyết:* nối seed entities và quan hệ trong các chunk có provenance, sau đó linearize thành context chung.
2. **Ca lỗi GraphRAG thất bại (hoặc cả hai cùng sai):**
   - *Question ID:* một câu `cross-doc` có seed không khớp exact.
   - *Nguyên nhân:* seed extraction/entity resolution có thể bỏ sót alias; corpus nhỏ không đại diện đầy đủ cho toàn bộ article dump.
   - *Đề xuất:* thêm alias map, ANN + lexical guard, hop-3 self-correction và chạy lại trên source dump đã được cấp quyền.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent
> **Trade-offs, Agent Control & Scale 350MB:** 
> - So sánh sự đánh đổi giữa GraphRAG vs Flat RAG về Latency, Token và Indexing Overhead.
> - Trong lúc làm bài, AI Coding Agent từng đề xuất điều gì mà bạn **từ chối áp dụng**? Tại sao?
> - Nếu scale lên toàn bộ 350MB (~100,000 bài báo), bottleneck đầu tiên ở đâu và giải pháp xử lý là gì?

*Trả lời:*
- **Đánh đổi:** GraphRAG tăng độ đầy đủ nhưng tốn indexing, traversal và token; benchmark local đo được graph 0.012s/36.42 tokens so với flat 0.011s/19.36 tokens.
- **Quyết định kỹ thuật:** giữ batch `UNWIND`, ANN top-k và cap context; tránh pairwise cosine O(N²) trên toàn bộ dataset.
- **Scale 350MB:** stream theo batch, lưu chunk ngoài RAM, dùng HNSW/FAISS, async extraction có retry, bulk Neo4j và community partitioning.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code
| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()` | Chỉ resolve khi antecedent rõ trong chunk |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Lọc quan hệ không hợp lệ |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | 109 edge, provenance invalid = 0 |
| **Entity Resolution & Union-Find** | Module 3 | `resolution_audit()` | 70 audit rows: manual, vector merge, guard rejection |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()` | Degree >100 → tối đa 50 edge |
| **Deterministic Evaluation** | Module 5 | `score()` | Token-overlap judge tái lập trên corpus nhỏ |

---

### 2. Quá trình Debugging & Bài học
- **Lỗi kỹ thuật phức tạp nhất gặp phải:** Phải giới hạn pipeline vào bộ first-5000-row được cung cấp thay vì tải toàn bộ dump.
- **Cách xử lý:** dùng detailed golden scope làm input bounded, giữ evidence/entity/relation metadata, kiểm tra Neo4j thật và provenance.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)
- **Tên đồ án / Dự án:** Trợ lý tri thức tài liệu kỹ thuật.
- **Đặc thù bài toán & Lý do chọn giải pháp:** dùng Hybrid RAG; GraphRAG cho câu hỏi quan hệ/phụ thuộc, vector retrieval cho mô tả dài và tài liệu không cấu trúc.
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `Document`, `Topic`, `Product`, `Person`, `Organization`.
  - Relations: `MENTIONS`, `DEPENDS_ON`, `AUTHORED_BY`, `IMPLEMENTS`, `RELATED_TO`.
- **Chiến lược:** alias map + ANN + lexical guard; super-node ưu tiên temporal/topical edges, cap 50 và global cap 250.

---

## 🎯 TỰ ĐÁNH GIÁ
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4/5 | Đã triển khai bulk graph, retrieval và cap policy |
| Khả năng kiểm soát AI Coding Agent | 4/5 | Có fallback và ghi rõ giới hạn dữ liệu |
| Chất lượng đồ thị tri thức xây dựng | 4/5 | 65 node, 109 edge, provenance invalid = 0 |
| Khả năng phân tích và debug hệ thống | 4/5 | Đã xác định gated dataset/model incompatibility |
