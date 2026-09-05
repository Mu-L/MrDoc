"""
MrDoc RAG 专用 Markdown Chunker（优化版）

特点：
- 基于结构（heading）切分（稳定）
- 自动构建 title_path（核心）
- 内置 chunk size 控制
- embedding 友好结构
- 支持后续 rerank / 多向量扩展
"""

from markdown_it import MarkdownIt
from markdownify import markdownify as html_to_markdown
import re


# =========================================================
# Markdown Parser
# =========================================================

# gfm-like 相比 commonmark 多了 table / strikethrough 规则，
# 否则表格不会被解析成 table_open，只能退化成一段管道原文。
md = MarkdownIt("gfm-like", {
    "html": False,
    "linkify": False,
    "typographer": False,
})


# =========================================================
# 文档归一化（HTML -> Markdown / 换行符统一）
# =========================================================

# 富文本编辑器（editor_mode=3）存的是 HTML，需要先转成 Markdown 才能解析出结构
_HTML_START_RE = re.compile(
    r"^\s*<(?:!doctype\s+html|html|body|div|p|h[1-6]|article|section|"
    r"table|thead|tbody|ul|ol|blockquote|pre|img|figure|span|font)\b",
    re.I,
)

_HTML_TAG_RE = re.compile(r"<\s*/?[a-zA-Z][a-zA-Z0-9\-]*(?:\s[^>]*)?>", re.I)

# markdown 文档里出现行首 # 标题时不做 HTML 转换，避免把正文标题降级成纯文本
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.M)

# 图片：data-uri / 超长 URL 会污染 embedding，需要占位；普通相对路径保留下来供答案引用
IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\s*\)"
)

MAX_IMAGE_SRC_LENGTH = 200


def collapse_long_images(text: str) -> str:
    """把 base64 内联图片 / 超长 src 图片替换为 [图:alt] 占位符。"""

    if not text or "![" not in text:
        return text

    def _replace(match):
        alt = match.group(1).strip()
        src = match.group(2).strip()
        if src.startswith("data:") or len(src) > MAX_IMAGE_SRC_LENGTH:
            return f"[图:{alt}]" if alt else "[图]"
        return match.group(0)

    return IMAGE_RE.sub(_replace, text)


def normalize_document_text(text) -> str:
    """切分前的归一化：类型兜底、换行统一、富文本 HTML 转 Markdown。"""

    if not isinstance(text, str):
        return ""

    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    looks_html = bool(_HTML_START_RE.match(text))
    if looks_html and len(_HTML_TAG_RE.findall(text)) >= 2:
        # 已经含 markdown 标题的混合内容不转换，保持原样
        if not _MD_HEADING_RE.search(text):
            converted = html_to_markdown(text, heading_style="ATX")
            if isinstance(converted, str) and converted.strip():
                return collapse_long_images(converted)

    return collapse_long_images(text)


# =========================================================
# 配置
# =========================================================

MAX_CHUNK_SIZE = 1200
MIN_MERGE_SIZE = 200

# =========================================================
# token 预算
# =========================================================
# 只按字符切块对中文是不够的：1200 个汉字约等于 1200 个 token，
# 而 embedding 模型都有输入上限，多出来的部分会在服务端被静默截断，
# 向量里根本看不到后半段内容。
# 该值必须与实际部署的 embedding 模型对齐：
# 若模型输入上限是 512（BGE / m3e / GTE 等），应下调到 512 - EMBED_META_TOKENS 以内；
# 大窗口模型（如 bge-m3，上限 8192）可保持 1024。
MAX_CHUNK_TOKENS = 1024

# embedding_text 除了正文还要拼文档标题、标签、章节路径等元信息，
# 装箱时先替这部分留出额度。
EMBED_META_TOKENS = 64

# 块间重叠：只带体积足够小的 block，避免把整个表格 / 代码块复制一遍
OVERLAP_MAX_CHARS = 120

# block 之间用 "\n\n" 拼接，装箱时要把这 2 个字符的开销算进去
BLOCK_JOIN_SEP = 2

# 结构拆小后还要按装箱口径复核，最多重拆几层，防止病态输入无限递归
SPLIT_MAX_DEPTH = 6


# 列表容器开闭 token 映射（嵌套遍历时需要配平）
LIST_PAIRS = {
    "bullet_list_open": "bullet_list_close",
    "ordered_list_open": "ordered_list_close",
}

LIST_CLOSE_TYPES = set(LIST_PAIRS.values())


# =========================================================
# 文本度量
# =========================================================

# 中日韩文字在主流 tokenizer 里基本一字一 token
CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3130-\u318f\u3400-\u4dbf"
    r"\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)


def estimate_token_count(text):
    """不依赖 tokenizer 的 token 估算：CJK 按字计，其余按 4 字符约 1 token。

    仓库里没有 tiktoken / transformers，embedding 模型又是运行时由
    AIProvider 决定的，拿不到真实词表，只能估算。
    对拉丁文本这个口径偏保守（实际往往 3 字符左右一个 token），
    宁可少装一点也不要让服务端静默截断。
    """

    if not text:

        return 0

    cjk = len(CJK_RE.findall(text))

    return cjk + (len(text) - cjk + 3) // 4


def char_room_for(chars, tokens, char_room, token_room):
    """把 token 余量折算成实际可用的字符上限。

    中英文密度差别很大：同样 448 的 token 余量，装汉字只够 448 个字符，
    装英文能塞 1700 多个。所以按待切分文本自身的密度折算，
    中文内容会自动切小，英文和代码仍然吃满字符上限。
    """

    if char_room <= 0:

        return 0

    if token_room <= 0 or tokens <= 0:

        return char_room

    token_room_chars = token_room * chars // tokens

    return max(min(char_room, token_room_chars), 1)


# =========================================================
# Markdown -> AST Block
# =========================================================

def markdown_to_blocks(text: str):

    tokens = md.parse(text)

    blocks = []

    # heading_stack 存标题，heading_levels 存对应层级
    # 裁剪时以已入栈的层级为准，标题跳级 / 回退都不会记错归属
    heading_stack = []

    heading_levels = []

    i = 0

    while i < len(tokens):

        token = tokens[i]

        # =====================================================
        # Heading：标题自身也产出 block，只有标题没有正文的小节不会消失
        # =====================================================

        if token.type == "heading_open":

            level = int(token.tag[1])

            title = collapse_long_images(
                inline_content(tokens, i)
            ).strip()

            if title:

                while heading_levels and heading_levels[-1] >= level:

                    heading_levels.pop()

                    heading_stack.pop()

                heading_levels.append(level)

                heading_stack.append(title)

                blocks.append({
                    "type": "heading",
                    "content": title,
                    "level": level,
                    "heading_path": heading_stack.copy(),
                })

            i += 3

            continue

        # =====================================================
        # Paragraph
        # =====================================================

        if token.type == "paragraph_open":

            content = collapse_long_images(
                inline_content(tokens, i)
            ).strip()

            if content:

                blocks.append({
                    "type": "paragraph",
                    "content": content,
                    "heading_path": heading_stack.copy(),
                })

            i += 3

            continue

        # =====================================================
        # Fence Code Block
        # =====================================================

        if token.type in ("fence", "code_block"):

            content = (token.content or "").strip()

            if content:

                blocks.append({
                    "type": "code",
                    "content": content,
                    "language": (token.info or "").strip(),
                    "heading_path": heading_stack.copy(),
                })

            i += 1

            continue

        # =====================================================
        # Table
        # =====================================================

        if token.type == "table_open":

            close = find_container_close(
                tokens,
                i,
                "table_open",
                "table_close"
            )

            table_content = extract_table_text(
                tokens[i + 1:close]
            )

            if table_content:

                blocks.append({
                    "type": "table",
                    "content": table_content,
                    "heading_path": heading_stack.copy(),
                })

            i = close + 1

            continue

        # =====================================================
        # List：一个列表项 = 一个 block，嵌套子项折叠进父项
        # =====================================================

        if token.type in LIST_PAIRS:

            close = find_container_close(
                tokens,
                i,
                token.type,
                LIST_PAIRS[token.type]
            )

            ordered = token.type == "ordered_list_open"

            for k, (item_beg, item_end) in enumerate(
                split_list_items(tokens, i + 1, close)
            ):

                marker = f"{k + 1}." if ordered else "-"

                item_content = collect_item_text(
                    tokens,
                    item_beg,
                    item_end,
                    "  "
                ).strip()

                if not item_content:

                    continue

                blocks.append({
                    "type": "list_item",
                    "content": f"{marker} {item_content}",
                    "heading_path": heading_stack.copy(),
                })

            i = close + 1

            continue

        # =====================================================
        # Blockquote：整段引用合成一个 block
        # =====================================================

        if token.type == "blockquote_open":

            close = find_container_close(
                tokens,
                i,
                "blockquote_open",
                "blockquote_close"
            )

            quote_content = collect_quote_text(
                tokens,
                i + 1,
                close
            ).strip()

            if quote_content:

                blocks.append({
                    "type": "quote",
                    "content": quote_content,
                    "heading_path": heading_stack.copy(),
                })

            i = close + 1

            continue

        i += 1

    return blocks


# =========================================================
# AST 容器遍历辅助
# =========================================================

def inline_content(tokens, index):
    """取 *_open 后紧跟的 inline 文本（标题 / 段落通用）。"""

    if index + 1 >= len(tokens):

        return ""

    token = tokens[index + 1]

    return token.content or "" if token.type == "inline" else ""


def find_container_close(tokens, start, open_type, close_type):
    """返回与 tokens[start] 配平的闭合 token 下标，找不到返回 len(tokens)。"""

    depth = 0

    for j in range(start, len(tokens)):

        t = tokens[j].type

        if t == open_type:

            depth += 1

        elif t == close_type:

            depth -= 1

            if depth == 0:

                return j

    return len(tokens)


def split_list_items(tokens, beg, end):
    """取列表区间内的直接子项，返回 [(inner_beg, inner_end), ...]（跳过嵌套列表）。"""

    items = []

    list_depth = 0

    inner_beg = None

    for j in range(beg, end):

        t = tokens[j].type

        if t in LIST_PAIRS:

            list_depth += 1

        elif t in LIST_CLOSE_TYPES:

            list_depth -= 1

        elif list_depth == 0 and t == "list_item_open":

            inner_beg = j + 1

        elif list_depth == 0 and t == "list_item_close":

            if inner_beg is not None:

                items.append((inner_beg, j))

            inner_beg = None

    return items


def collect_item_text(tokens, beg, end, indent=""):
    """把一个 list_item 内部内容收集成文本。

    约定：返回文本的首行不带缩进（由上层拼 "- " 前缀），其余行携带绝对缩进，
    这样嵌套列表的层级关系能在正文里保留下来。
    """

    out = []

    first = True

    def add_block(text):

        nonlocal first

        for line in (text or "").split("\n"):

            if not line.strip():

                continue

            if first:

                out.append(line.strip())

                first = False

            else:

                out.append(indent + line.strip())

    i = beg

    while i < end:

        t = tokens[i].type

        if t == "paragraph_open":

            add_block(collapse_long_images(inline_content(tokens, i)))

            i += 3

        elif t in LIST_PAIRS:

            close = find_container_close(tokens, i, t, LIST_PAIRS[t])

            ordered = t == "ordered_list_open"

            for k, (sb, se) in enumerate(split_list_items(tokens, i + 1, close)):

                marker = f"{k + 1}." if ordered else "-"

                child = collect_item_text(tokens, sb, se, indent + "  ")

                if not child.strip():

                    continue

                # 子项首行补上本层缩进与列表标记
                lines = child.split("\n")

                head = lines[0].strip()

                if first:

                    out.append(f"{marker} {head}")

                    first = False

                else:

                    out.append(f"{indent}{marker} {head}")

                out.extend(lines[1:])

            i = close + 1

        elif t == "blockquote_open":

            close = find_container_close(
                tokens, i, "blockquote_open", "blockquote_close"
            )

            add_block(collect_quote_text(tokens, i + 1, close))

            i = close + 1

        elif t in ("fence", "code_block"):

            add_block(tokens[i].content)

            i += 1

        elif t == "table_open":

            close = find_container_close(tokens, i, "table_open", "table_close")

            add_block(extract_table_text(tokens[i + 1:close]))

            i = close + 1

        else:

            i += 1

    return "\n".join(out)


def collect_quote_text(tokens, beg, end):
    """把 blockquote 内部内容收集成纯文本（不带 > 前缀）。"""

    parts = []

    i = beg

    while i < end:

        t = tokens[i].type

        if t == "paragraph_open":

            text = collapse_long_images(inline_content(tokens, i)).strip()

            if text:

                parts.append(text)

            i += 3

        elif t in LIST_PAIRS:

            close = find_container_close(tokens, i, t, LIST_PAIRS[t])

            ordered = t == "ordered_list_open"

            lines = []

            for k, (sb, se) in enumerate(split_list_items(tokens, i + 1, close)):

                marker = f"{k + 1}." if ordered else "-"

                body = collect_item_text(tokens, sb, se, "  ").strip()

                if body:

                    lines.append(f"{marker} {body}")

            if lines:

                parts.append("\n".join(lines))

            i = close + 1

        elif t == "blockquote_open":

            close = find_container_close(
                tokens, i, "blockquote_open", "blockquote_close"
            )

            body = collect_quote_text(tokens, i + 1, close).strip()

            if body:

                parts.append(
                    "\n".join(f"> {line}" for line in body.split("\n"))
                )

            i = close + 1

        elif t in ("fence", "code_block"):

            content = (tokens[i].content or "").strip()

            if content:

                parts.append(content)

            i += 1

        else:

            i += 1

    return "\n".join(parts)


def _plain_text_blocks(text: str):
    """兜底：AST 解析不出任何 block 时按空行切段，保证仍有 chunk 产出。"""

    blocks = []

    for paragraph in re.split(r"\n\s*\n", text.strip()):

        content = collapse_long_images(paragraph).strip()

        if not content:

            continue

        blocks.append({
            "type": "paragraph",
            "content": content,
            "heading_path": [],
        })

    return blocks


# =========================================================
# 提取表格文本
# =========================================================

def extract_table_text(tokens):

    rows = []

    current_row = []

    for token in tokens:

        if token.type == "inline":

            current_row.append(
                collapse_long_images(token.content).strip()
            )

        elif token.type == "tr_close":

            if current_row:

                rows.append(current_row)

            current_row = []

    return "\n".join([
        " | ".join(row)
        for row in rows
    ])


# =========================================================
# block -> 文本
# =========================================================

def render_block(block):
    """原文口径渲染：用于 chunk 的 content 字段，同时也是装箱度量的基准。"""

    block_type = block["type"]

    # =====================================================
    # heading：保留 ATX 前缀，让 embedding 能感知层级
    # =====================================================

    if block_type == "heading":

        level = int(block.get("level") or 1)

        return "#" * min(max(level, 1), 6) + f" {block['content']}"

    # =====================================================
    # paragraph
    # =====================================================

    if block_type == "paragraph":

        return block["content"]

    # =====================================================
    # code
    # =====================================================

    if block_type == "code":

        lang = block.get("language", "")

        if lang:

            return (
                f"[代码块:{lang}]\n"
                f"{block['content']}"
            )

        return (
            "[代码块]\n"
            f"{block['content']}"
        )

    # =====================================================
    # table
    # =====================================================

    if block_type == "table":

        return (
            "[表格]\n"
            f"{block['content']}"
        )

    return block.get("content", "")


# =========================================================
# 表格 -> 自然语言
# =========================================================

# 表格单元格分隔符；转义过的 \| 属于内容，不参与切分
TABLE_CELL_RE = re.compile(r"(?<!\\)\|")


def split_table_row(line):
    return [cell.strip() for cell in TABLE_CELL_RE.split(line)]


def verbalize_table_row(header, cells):
    """一行表格渲染成 "列名：值；列名：值"。

    裸的 "值1 | 值2 | 值3" 对模型没有语义，向量里检索不到列名，
    带上列名之后 "超时时间是多少" 这类问题才能命中对应行。
    """

    pairs = []

    for index, cell in enumerate(cells):

        if not cell:

            continue

        name = header[index] if index < len(header) else ""

        pairs.append(f"{name}：{cell}" if name else cell)

    return "；".join(pairs)


def verbalize_table(content):
    """把 "列 | 列" 形式的表格文本整体转成自然语言，逐行一句。"""

    lines = [
        line.strip()
        for line in (content or "").split("\n")
        if line.strip()
    ]

    if not lines:

        return ""

    header = split_table_row(lines[0])

    # 只有表头（或被拆得只剩表头）时直接报列名
    if len(lines) == 1:

        return "；".join(cell for cell in header if cell)

    rows = [
        verbalize_table_row(header, split_table_row(line))
        for line in lines[1:]
    ]

    return "\n".join(row for row in rows if row)


def render_block_for_embedding(block):
    """向量化口径：表格换成自然语言，其余类型与原文口径一致。"""

    if block["type"] != "table":

        return render_block(block)

    rows = verbalize_table(block.get("content"))

    if not rows:

        return ""

    return f"[表格]\n{rows}"


def block_size(block):
    """block 在两种口径下的体积：(字符数, token 数)。

    content 走原文口径、embedding_text 走向量化口径，
    两边都不能超限，所以各取两个渲染结果里更大的那个。
    """

    display = render_block(block)

    embedded = render_block_for_embedding(block)

    return (
        max(len(display), len(embedded)),
        max(
            estimate_token_count(display),
            estimate_token_count(embedded),
        ),
    )


# =========================================================
# semantic merge
# =========================================================

def semantic_merge_blocks(
    blocks,
    min_merge_size=MIN_MERGE_SIZE
):

    merged = []

    buffer = []

    buffer_length = 0

    current_heading = None

    for block in blocks:

        heading = tuple(block["heading_path"])

        rendered = render_block(
            block
        )

        size = len(rendered)

        # heading 变化
        if (
            current_heading is not None
            and heading != current_heading
            and buffer_length >= min_merge_size
        ):

            merged.append(buffer)

            buffer = []

            buffer_length = 0

        buffer.append(block)

        buffer_length += size

        current_heading = heading

    if buffer:

        merged.append(buffer)

    return merged


# =========================================================
# semantic split
# =========================================================

def hard_cut_text(text, limit):
    """没有任何断点时的最后兜底：按字符硬切，保证每段都不超限。"""

    if limit <= 0:

        return [text]

    if len(text) <= limit:

        return [text]

    return [
        text[i:i + limit]
        for i in range(0, len(text), limit)
    ]


def split_large_text(
    text,
    char_room,
    token_room
):

    sentences = re.split(
        r'(?<=[。！？；.!?;])',
        text
    )

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    chunks = []

    current = ""

    for sentence in sentences:

        sentence_limit = char_room_for(
            len(sentence),
            estimate_token_count(sentence),
            char_room,
            token_room
        )

        # 单句本身超限（无标点的一整段）时退化为硬切
        if len(sentence) > sentence_limit:

            if current:

                chunks.append(current)

                current = ""

            chunks.extend(
                hard_cut_text(sentence, sentence_limit)
            )

            continue

        joined = current + sentence

        if (
            len(joined) <= char_room and
            estimate_token_count(joined) <= token_room
        ):

            current = joined

        else:

            if current:

                chunks.append(current)

            current = sentence

    if current:

        chunks.append(current)

    return chunks


def pack_lines(lines, limit, costs=None):
    """按行打包，每段长度（含换行）不超过 limit。

    costs 与 lines 一一对应时按指定体积记账，
    表格行在 embedding 里会被渲染成自然语言，比原文更长，
    装箱时得按更大的那个口径算。
    """

    if limit <= 0:

        limit = 1

    if costs is None:

        costs = [len(line) for line in lines]

    pieces = []

    current = []

    size = 0

    for index, line in enumerate(lines):

        cost = max(costs[index], 0)

        # 行本身装不下时按比例决定硬切的字符数，
        # 否则一段的体积会按 cost 记超上限
        cut = limit

        if cost > len(line) and cost > limit and line:

            cut = max(1, limit * len(line) // cost)

        parts = hard_cut_text(line, cut)

        for part in parts:

            part_cost = cost

            if len(parts) > 1 and line:

                part_cost = -(-cost * len(part) // len(line))

            extra = part_cost + (1 if current else 0)

            if current and size + extra > limit:

                pieces.append("\n".join(current))

                current = []

                size = 0

                extra = part_cost

            current.append(part)

            size += extra

    if current:

        pieces.append("\n".join(current))

    return pieces


def split_large_block(block, char_room, token_room):
    """把超限 block 拆成若干不超限的子 block。

    code / table 按行拆（表格每段都带上表头），其余按句子拆，
    句子仍然超长时由 split_large_text 内部按字符硬切。
    """

    chars, tokens = block_size(block)

    if chars <= char_room and tokens <= token_room:

        return [block]

    # 装箱时是按两种口径里更大的那个判断能不能放下，
    # 折算字符上限也要用同一个口径，
    # 否则切出来的段落会在主循环里被判定成放不下
    limit = char_room_for(chars, tokens, char_room, token_room)

    block_type = block["type"]

    heading_path = block.get("heading_path", [])

    # =====================================================
    # table：保留表头
    # =====================================================

    if block_type == "table":

        lines = [
            line
            for line in (block.get("content") or "").split("\n")
            if line.strip()
        ]

        if not lines:

            return []

        header = lines[0]

        body = lines[1:]

        reserve = len("[表格]\n")

        # 表头本身已经放不下时不再重复，否则每段都会比上限还长
        repeat_header = bool(body) and (limit - reserve - len(header) - 1) > 0

        if repeat_header:

            reserve += len(header) + 1

        rows = body or [header]

        header_cells = split_table_row(header)

        # 行渲染成自然语言后可能比原文更长，装箱按大的那个口径记
        costs = [
            max(
                len(row),
                len(verbalize_table_row(
                    header_cells,
                    split_table_row(row)
                )),
            )
            for row in rows
        ]

        out = []

        for piece in pack_lines(rows, limit - reserve, costs):

            content = piece

            if repeat_header:

                content = f"{header}\n{piece}"

            out.append({
                "type": "table",
                "content": content,
                "heading_path": heading_path,
            })

        return out

    # =====================================================
    # code：按行拆，保留语言标记
    # =====================================================

    if block_type == "code":

        language = (block.get("language") or "").strip()

        prefix = (
            f"[代码块:{language}]\n"
            if language
            else "[代码块]\n"
        )

        return [
            {
                "type": "code",
                "content": piece,
                "language": language,
                "heading_path": heading_path,
            }
            for piece in pack_lines(
                (block.get("content") or "").split("\n"),
                limit - len(prefix)
            )
        ]

    # =====================================================
    # paragraph / list_item / quote / heading
    # =====================================================

    return [
        {
            "type": "paragraph",
            "content": piece,
            "heading_path": heading_path,
        }
        for piece in split_large_text(
            render_block(block),
            limit,
            token_room
        )
    ]


def iter_fit_blocks(blocks, char_room, token_room, depth=0):
    """产出装箱用的 block，超限的先按结构拆小。

    split_large_block 折算字符上限时用的是整块的平均密度，
    而行与行之间的中英文比例可能差很多，拆出来的一段仍然可能顶破余量。
    这里用和装箱一致的口径复核一遍，超了就换更小的余量再拆一次；
    再拆也不会有进展时直接放行，交给装箱兜底。
    """

    for block in blocks:

        for fitted in split_large_block(block, char_room, token_room):

            chars, tokens = block_size(fitted)

            if chars <= char_room and tokens <= token_room:

                yield fitted

                continue

            next_chars = min(char_room, chars - 1)

            next_tokens = min(token_room, tokens - 1)

            progressed = (
                next_chars < char_room or
                next_tokens < token_room
            )

            if depth < SPLIT_MAX_DEPTH and progressed:

                yield from iter_fit_blocks(
                    [fitted],
                    next_chars,
                    next_tokens,
                    depth + 1
                )

                continue

            yield fitted


def measure_blocks(blocks):
    """一组 block 拼起来后的体积：(字符数, token 数)。

    block 之间用 "\n\n" 拼接，字符数记 2，token 数按 2 估（实际约 1，
    取大值只是为了不低估）。
    """

    if not blocks:

        return (0, 0)

    chars = 0

    tokens = 0

    for block in blocks:

        size = block_size(block)

        chars += size[0]

        tokens += size[1]

    join = BLOCK_JOIN_SEP * (len(blocks) - 1)

    return (chars + join, tokens + join)


def pick_overlap_blocks(blocks, char_budget, token_budget):
    """取尾部若干小块作为块间重叠，太大的 block 不带，避免整块复制。"""

    overlap = []

    chars = 0

    tokens = 0

    for block in reversed(blocks):

        size = block_size(block)

        if (
            chars + size[0] + BLOCK_JOIN_SEP > char_budget or
            tokens + size[1] + BLOCK_JOIN_SEP > token_budget
        ):

            break

        overlap.insert(0, block)

        chars += size[0] + BLOCK_JOIN_SEP

        tokens += size[1] + BLOCK_JOIN_SEP

    return overlap


# =========================================================
# build chunk
# =========================================================

def common_heading_path(blocks):
    """取一组 block 的章节公共前缀。

    小组（不足 MIN_MERGE_SIZE）会跨标题合并，这时组内 block 的 heading_path
    并不相同；取公共前缀可以避免把整个 chunk 归属到其中某一个小节下。

    文档开头的无标题前言（heading_path 为空）不参与求值：这些游离块不在任何
    章节下，若与后续标题块同 chunk，空路径会把公共前缀清空，
    导致 chunk 丢失本应归属的章节标题（title_path / section_title 变空）。
    """

    paths = [
        block.get("heading_path") or []
        for block in blocks
    ]

    paths = [p for p in paths if p]

    if not paths:

        return []

    prefix = list(paths[0])

    for path in paths[1:]:

        depth = 0

        while (
            depth < len(prefix)
            and depth < len(path)
            and prefix[depth] == path[depth]
        ):

            depth += 1

        prefix = prefix[:depth]

        if not prefix:

            break

    return prefix or []


def build_chunk(
    blocks,
    doc_title,
    doc_tags
):

    heading_path = []

    title_path = ""

    section_title = ""

    # =====================================================
    # heading 信息
    # =====================================================

    if blocks:

        heading_path = common_heading_path(blocks)

        title_path = " > ".join(
            heading_path
        )

        if heading_path:

            section_title = heading_path[-1]

        elif doc_title:

            # 首个标题之前的前言：归属到文档标题，避免出现空路径
            title_path = doc_title

            section_title = doc_title

        # DocAISection 的 title_path / section_title 是 varchar(1024)，
        # 超长标题必须截断，否则整篇文档索引会因插入失败而中断
        if len(title_path) > 1024:

            title_path = title_path[:1024]

        if len(section_title) > 1024:

            section_title = section_title[:1024]

    # =====================================================
    # 渲染 block：content 保留原文，embedding_text 用向量化口径
    # =====================================================

    display_blocks = []

    embed_blocks = []

    for block in blocks:

        display_blocks.append(render_block(block))

        embed_blocks.append(render_block_for_embedding(block))

    content = "\n\n".join(display_blocks)

    embedding_content = "\n\n".join(embed_blocks)

    # =====================================================
    # embedding text
    # =====================================================

    embedding_text = f"""
文档标题: {doc_title}

文档标签：{doc_tags}

章节路径: {title_path}

章节标题: {section_title}

内容:
{embedding_content}
""".strip()

    # =====================================================
    # 返回 chunk
    # =====================================================

    return {

        # chunk 内容
        "content": content,

        # =================================================
        # 兼容旧版结构
        # =================================================

        "section_title": section_title,

        "title_path": title_path,

        # embedding
        "embedding_text": embedding_text
    }


# =========================================================
# 主 chunker
# =========================================================

def chunk_markdown_document(
    text,
    doc_title="",
    doc_tags="",
    max_chunk_size=MAX_CHUNK_SIZE
):

    # =====================================================
    # 归一化：None / 非字符串 / CRLF / 富文本 HTML
    # =====================================================

    text = normalize_document_text(text)

    if not text.strip():

        return []

    # =====================================================
    # markdown -> semantic blocks
    # =====================================================

    blocks = markdown_to_blocks(text)

    # 解析不出结构时降级为纯文本切分，避免返回空结果导致调用方清空索引
    if not blocks:

        blocks = _plain_text_blocks(text)

    if not blocks:

        return []

    # =====================================================
    # semantic merge
    # =====================================================

    merged_groups = semantic_merge_blocks(
        blocks
    )

    chunks = []

    # 正文可用的 token 余量：embedding_text 还要拼元信息，先扣掉预留额度
    token_budget = max(
        MAX_CHUNK_TOKENS - EMBED_META_TOKENS,
        32
    )

    for group in merged_groups:

        current_blocks = []

        current_chars = 0

        current_tokens = 0

        # 超长的 code / table / 段落先按结构拆小，保证装箱不会越界
        for raw_block in group:

            char_room = max_chunk_size

            room_tokens = token_budget

            # 组里还只攒着标题时拆得稍小一点，
            # 让标题能和它领起的内容落在同一个 chunk 里
            if (
                current_blocks and
                all(
                    b["type"] == "heading"
                    for b in current_blocks
                )
            ):

                char_room = max(
                    max_chunk_size - current_chars - BLOCK_JOIN_SEP,
                    max_chunk_size // 2
                )

                room_tokens = max(
                    token_budget - current_tokens - BLOCK_JOIN_SEP,
                    token_budget // 2
                )

            for block in iter_fit_blocks(
                [raw_block],
                char_room,
                room_tokens
            ):

                size = block_size(block)

                separator = BLOCK_JOIN_SEP if current_blocks else 0

                # =================================================
                # 放得下：继续装箱
                # =================================================

                if (
                    current_chars + separator + size[0] <= max_chunk_size and
                    current_tokens + separator + size[1] <= token_budget
                ):

                    current_blocks.append(block)

                    current_chars += separator + size[0]

                    current_tokens += separator + size[1]

                    continue

                # =================================================
                # 放不下：先收尾，再带一点小块重叠
                # =================================================

                chunks.append(
                    build_chunk(
                        current_blocks,
                        doc_title,
                        doc_tags
                    )
                )

                # 只在新块留出的空间里放重叠，放不下就少放，绝不顶破上限
                overlap = pick_overlap_blocks(
                    current_blocks,
                    min(
                        OVERLAP_MAX_CHARS,
                        max(max_chunk_size - size[0] - BLOCK_JOIN_SEP, 0)
                    ),
                    max(token_budget - size[1] - BLOCK_JOIN_SEP, 0)
                )

                current_blocks = overlap + [block]

                current_chars, current_tokens = measure_blocks(
                    current_blocks
                )

        # =====================================================
        # group 收尾
        # =====================================================

        if current_blocks:

            chunks.append(
                build_chunk(
                    current_blocks,
                    doc_title,
                    doc_tags
                )
            )

    return chunks
