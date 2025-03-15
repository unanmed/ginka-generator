import torch
from ..model.model import DynamicPadConv, ConditionInjector, HybridUpsample

def test_dynamic_conv():
    conv = DynamicPadConv(3, 64, stride=2)
    
    # 测试奇数尺寸
    x = torch.randn(1, 3, 15, 17)
    out = conv(x)
    assert out.shape == (1, 64, 8, 9), f"Got {out.shape}"
    
    # 测试偶数尺寸
    x = torch.randn(1, 3, 16, 16)
    out = conv(x)
    assert out.shape == (1, 64, 8, 8)
    
def test_condition_injector():
    injector = ConditionInjector(128, 256)
    x = torch.randn(2, 256, 16, 16)
    cond = torch.randn(2, 128)
    
    out = injector(x, cond)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)  # 确保条件起作用

def test_hybrid_upsample():
    # 带跳跃连接的情况
    upsample = HybridUpsample(256, 128, skip_ch=64)
    x = torch.randn(2, 256, 8, 8)
    skip = torch.randn(2, 64, 16, 16)
    out = upsample(x, skip)
    assert out.shape == (2, 128, 16, 16)
    
    # 无跳跃连接的情况
    upsample = HybridUpsample(256, 128)
    out = upsample(x)
    assert out.shape == (2, 128, 16, 16)

def test_all():
    test_dynamic_conv()
    print("✅ 动态卷积测试完毕")
    test_condition_injector()
    print("✅ 条件注入测试完毕")
    test_hybrid_upsample()
    print("✅ 混合上采样测试完毕")
    