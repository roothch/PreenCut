"""TwelveLabs 分段后端的测试。

无网络的解析单测可直接运行：
    python -m tests.test_twelvelabs_processor
或使用 pytest：
    pytest tests/test_twelvelabs_processor.py

设置 TWELVELABS_API_KEY 后会额外跑一个 Marengo 检索能力的连通性冒烟测试，
未设置时自动跳过。
"""
import os
import sys

# 允许从仓库根目录直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.twelvelabs_processor import (
    _strip_code_fence,
    _overlaps_any,
    _parse_segments,
)


def test_strip_code_fence():
    assert _strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_code_fence('```\n[]\n```') == '[]'
    assert _strip_code_fence('{"a": 1}') == '{"a": 1}'


def test_overlaps_any():
    windows = [(0.0, 5.0), (10.0, 12.0)]
    assert _overlaps_any(1.0, 3.0, windows) is True    # 完全落入第一个窗口
    assert _overlaps_any(4.0, 11.0, windows) is True   # 跨越两个窗口
    assert _overlaps_any(6.0, 9.0, windows) is False   # 落在窗口之间
    assert _overlaps_any(1.0, 3.0, []) is False        # 无窗口


def test_parse_segments():
    raw = (
        '[{"start": 0, "end": 5.5, "summary": "开场", "tags": ["介绍"]},'
        ' {"start": 5.5, "end": 10, "summary": "演示", "tags": "产品"}]'
    )
    segs = _parse_segments(raw)
    assert len(segs) == 2
    assert segs[0]["start"] == 0.0 and segs[0]["end"] == 5.5
    assert segs[1]["tags"] == ["产品"]  # 字符串标签归一化为列表


def test_parse_segments_invalid_raises():
    try:
        _parse_segments("not json at all")
    except ValueError:
        pass
    else:
        raise AssertionError("无效输入应抛出 ValueError")


def test_marengo_smoke():
    """连通性冒烟测试：Marengo 文本嵌入返回向量。需要 TWELVELABS_API_KEY。"""
    if not os.getenv("TWELVELABS_API_KEY"):
        print("SKIP test_marengo_smoke: TWELVELABS_API_KEY 未设置")
        return
    from twelvelabs import TwelveLabs
    client = TwelveLabs(api_key=os.environ["TWELVELABS_API_KEY"])
    emb = client.embed.create(model_name="marengo3.0", text="a cat playing piano")
    vector = emb.text_embedding.segments[0].float_
    assert len(vector) > 0
    print(f"OK test_marengo_smoke: 向量维度={len(vector)}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("全部测试通过")
